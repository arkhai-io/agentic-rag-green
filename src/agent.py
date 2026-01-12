"""Green Agent logic for RAG Assessment.

This module contains the main agent logic that processes Purple Agent requests
and coordinates with the RAG environment.

The RAGAssessorAgent:
1. Receives A2A messages from Purple Agents
2. Parses action requests (register, create_pipeline, query, etc.)
3. Delegates to the RAGEnvironment for execution
4. Returns results via A2A protocol

Architecture:
------------
    A2A Message (from Purple Agent)
           │
           ▼
    ┌─────────────────────────────────┐
    │      RAGAssessorAgent           │
    │                                 │
    │  run():                         │
    │  1. Parse message as JSON       │
    │  2. Extract action + params     │
    │  3. Dispatch to environment     │
    │  4. Format and return response  │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │      RAGEnvironment             │
    │                                 │
    │  • register_agent()             │
    │  • create_project()             │
    │  • create_pipeline()            │
    │  • query()                      │
    └─────────────────────────────────┘
"""

import json
from typing import Any, Dict, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from pydantic import ValidationError

from .environment import RAGEnvironment
from .messenger import Messenger
from .models import ActionRequest, ActionResponse, ActionType, EvalRequest


# =============================================================================
# GREEN AGENT CLASS
# =============================================================================

class RAGAssessorAgent:
    """
    Main Green Agent for RAG assessment/environment.
    
    This agent serves two roles:
    
    1. **Environment Provider**: Provides a RAG environment for Purple Agents
       to interact with (registration, pipeline creation, queries, etc.)
    
    2. **Assessor**: When running a full assessment, coordinates tasks
       and evaluates Purple Agent performance.
    
    The agent maintains:
    - environment: RAGEnvironment instance for executing operations
    - messenger: Messenger for A2A communication with other agents
    - current_agent_name: The registered name of the Purple Agent (per session)
    
    Message Format:
    ---------------
    Purple Agents send JSON messages with the following structure:
    
        {
            "action": "<action_type>",
            "params": { ... action-specific params ... }
        }
    
    Supported actions:
    - register: Register a new agent (first action required)
    - create_project: Create a new project
    - create_pipeline: Create a RAG pipeline
    - list_projects: List agent's projects
    - list_pipelines: List agent's pipelines
    - query: Run a RAG query
    
    Example Usage:
    --------------
    # Purple Agent sends registration request:
    {
        "action": "register",
        "params": {"agent_name": "my_rag_agent"}
    }
    
    # Green Agent responds:
    {
        "success": true,
        "action": "register",
        "data": {
            "status": "success",
            "agent_name": "my_rag_agent",
            "message": "Agent registered successfully"
        }
    }
    """

    # Required roles when running a full assessment (from AgentBeats)
    required_roles: list[str] = ["rag_agent"]
    
    # Required config keys for assessment
    required_config_keys: list[str] = []

    def __init__(self):
        """Initialize the Green Agent."""
        self.environment = RAGEnvironment()
        self.messenger = Messenger()
        
        # Track the registered agent name for this session
        self.current_agent_name: Optional[str] = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Process an incoming A2A message.
        
        This is the main entry point called by the Executor. It:
        1. Extracts the message text
        2. Parses it as JSON to get the action request
        3. Dispatches to the appropriate handler
        4. Reports the result via the updater
        
        Args:
            message: The incoming A2A message
            updater: TaskUpdater for reporting progress and results
        
        Message Handling:
        ----------------
        The message is expected to be a JSON string with:
        - action: The action type (register, create_pipeline, etc.)
        - params: Action-specific parameters
        
        If parsing fails or the action is invalid, an error response is sent.
        """
        input_text = get_message_text(message)
        
        # Try to parse as ActionRequest (Purple Agent action)
        try:
            # First, try to parse as raw JSON
            data = json.loads(input_text)
            
            # Check if it's an EvalRequest (from AgentBeats platform)
            if "participants" in data and "config" in data:
                await self._handle_assessment_request(data, updater)
                return
            
            # Otherwise, parse as ActionRequest (from Purple Agent)
            action_request = ActionRequest(**data)
            await self._handle_action_request(action_request, updater)
            
        except json.JSONDecodeError as e:
            # Not valid JSON - might be plain text
            await self._handle_text_message(input_text, updater)
            
        except ValidationError as e:
            # Valid JSON but invalid ActionRequest structure
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message(
                    json.dumps({
                        "success": False,
                        "error": f"Invalid request format: {e}",
                        "expected_format": {
                            "action": "<action_type>",
                            "params": {"...": "..."}
                        }
                    })
                )
            )

    async def _handle_action_request(
        self, 
        request: ActionRequest, 
        updater: TaskUpdater
    ) -> None:
        """
        Handle an action request from a Purple Agent.
        
        Routes the request to the appropriate handler based on action type.
        
        Args:
            request: The parsed ActionRequest
            updater: TaskUpdater for reporting results
        """
        # Ensure environment is initialized
        await self.environment.initialize()
        
        # Update status to show we're working
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Processing action: {request.action.value}")
        )
        
        # Special handling for registration (sets current_agent_name)
        if request.action == ActionType.REGISTER:
            response = await self.environment.handle_action(
                action=request.action,
                params=request.params,
            )
            
            # Set session for both success and already_exists (allows session restore after restart)
            if response.data:
                status = response.data.get("status")
                if status in ("success", "already_exists"):
                    self.current_agent_name = response.data.get("agent_name")
                    # Mark as success for already_exists (session restored)
                    if status == "already_exists":
                        response.success = True
                        response.error = None
                        response.data["message"] = f"Session restored for agent '{self.current_agent_name}'."
        
        else:
            # For other actions, use the current agent name
            if not self.current_agent_name:
                response = ActionResponse(
                    success=False,
                    action=request.action,
                    error="Not registered. Please register first with action='register'.",
                )
            else:
                response = await self.environment.handle_action(
                    action=request.action,
                    params=request.params,
                    agent_name=self.current_agent_name,
                )
        
        # Send response
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(response.model_dump_json())
        )

    async def _handle_assessment_request(
        self, 
        data: Dict[str, Any], 
        updater: TaskUpdater
    ) -> None:
        """
        Handle an assessment request from the AgentBeats platform.
        
        This is called when the Green Agent receives an EvalRequest
        to run a full assessment on a Purple Agent.
        
        Args:
            data: The parsed EvalRequest data
            updater: TaskUpdater for reporting results
        """
        try:
            eval_request = EvalRequest(**data)
            
            # Validate required roles
            missing_roles = set(self.required_roles) - set(eval_request.participants.keys())
            if missing_roles:
                await updater.reject(
                    new_agent_text_message(f"Missing required roles: {missing_roles}")
                )
                return
            
            # Validate required config
            missing_config = set(self.required_config_keys) - set(eval_request.config.keys())
            if missing_config:
                await updater.reject(
                    new_agent_text_message(f"Missing required config: {missing_config}")
                )
                return
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Starting assessment...")
            )
            
            # TODO: Implement full assessment logic
            # This would:
            # 1. Load benchmark tasks
            # 2. Send tasks to Purple Agent via A2A
            # 3. Evaluate responses
            # 4. Report scores
            
            # For now, just acknowledge the assessment request
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text="Assessment acknowledged")),
                    Part(root=DataPart(data={
                        "participants": {k: str(v) for k, v in eval_request.participants.items()},
                        "config": eval_request.config,
                        "status": "not_yet_implemented",
                    }))
                ],
                name="Assessment Result",
            )
            
        except ValidationError as e:
            await updater.reject(
                new_agent_text_message(f"Invalid EvalRequest: {e}")
            )

    async def _handle_text_message(
        self, 
        text: str, 
        updater: TaskUpdater
    ) -> None:
        """
        Handle a plain text message (not JSON).
        
        Provides helpful guidance on the expected message format.
        
        Args:
            text: The plain text message
            updater: TaskUpdater for reporting results
        """
        help_message = {
            "message": "I expect JSON action requests. Here's the format:",
            "format": {
                "action": "<action_type>",
                "params": {"...": "..."}
            },
            "available_actions": [action.value for action in ActionType],
            "examples": {
                "register": {
                    "action": "register",
                    "params": {"agent_name": "my_agent"}
                },
                "create_project": {
                    "action": "create_project",
                    "params": {"project_name": "my_project"}
                },
                "create_pipeline": {
                    "action": "create_pipeline",
                    "params": {
                        "pipeline_name": "my_pipeline",
                        "project_name": "my_project",
                        "components": [
                            {"type": "chunker", "name": "sentence_chunker"},
                            {"type": "embedder", "name": "openai_embedder"}
                        ],
                        "pipeline_type": "indexing"
                    }
                }
            },
            "received_text": text[:200] + ("..." if len(text) > 200 else "")
        }
        
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(json.dumps(help_message, indent=2))
        )
