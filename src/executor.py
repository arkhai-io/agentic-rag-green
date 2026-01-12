"""A2A Request Handler (Executor) for the Green Agent.

This module implements the A2A protocol's AgentExecutor interface,
handling incoming requests from Purple Agents and the AgentBeats platform.

The Executor:
1. Receives A2A messages
2. Parses the action request (registration, pipeline ops, queries)
3. Routes to the appropriate handler in the Agent
4. Returns results via A2A protocol

A2A Flow:
---------
    Incoming A2A Message
           │
           ▼
    ┌──────────────┐
    │   Executor   │  ◄── Implements AgentExecutor interface
    │              │
    │ • Validates  │
    │ • Routes     │
    │ • Updates    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │    Agent     │  ◄── Business logic (environment operations)
    │              │
    │ • Register   │
    │ • Pipeline   │
    │ • Query      │
    └──────────────┘
"""

from typing import Dict, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from .agent import RAGAssessorAgent


# =============================================================================
# CONSTANTS
# =============================================================================

# Terminal task states - once in these states, task cannot be modified
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


# =============================================================================
# EXECUTOR CLASS
# =============================================================================

class Executor(AgentExecutor):
    """
    A2A request handler that routes messages to the Green Agent.
    
    This class implements the A2A protocol's AgentExecutor interface.
    It's responsible for:
    
    1. **Message Validation**: Ensuring incoming messages are properly formatted
    2. **Task Management**: Creating and tracking A2A tasks
    3. **Agent Dispatch**: Routing requests to the RAGAssessorAgent
    4. **Response Handling**: Converting agent results to A2A responses
    
    The Executor maintains agent instances per context_id, allowing
    multiple concurrent assessments with isolated state.
    
    Attributes:
        agents: Dict mapping context_id to RAGAssessorAgent instances
    
    Example flow:
        1. Purple Agent sends: {"action": "register", "params": {"agent_name": "test"}}
        2. Executor receives message, creates task
        3. Executor calls agent.run(message, updater)
        4. Agent processes registration, calls updater.complete()
        5. Executor returns result via A2A protocol
    """

    def __init__(self):
        """Initialize executor with empty agent tracking."""
        # Map context_id -> agent instance for isolation
        self.agents: Dict[str, RAGAssessorAgent] = {}

    async def execute(
        self, 
        context: RequestContext, 
        event_queue: EventQueue
    ) -> None:
        """
        Execute an incoming A2A request.
        
        This is the main entry point for all A2A messages. It:
        1. Validates the request has a message
        2. Creates or retrieves the task
        3. Gets or creates an agent instance for this context
        4. Delegates to the agent's run() method
        5. Handles completion/failure
        
        Args:
            context: A2A request context containing the message and task info
            event_queue: Queue for sending A2A events (updates, artifacts)
        
        Raises:
            ServerError: If the request is invalid or task is already processed
        """
        # Validate message exists
        msg = context.message
        if not msg:
            raise ServerError(
                error=InvalidRequestError(message="Missing message in request")
            )

        # Check if task already exists and is in terminal state
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        # Create new task if needed
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        # Get or create agent instance for this context
        # Each context_id gets its own agent instance for isolation
        context_id = task.context_id
        print(f"[DEBUG] context_id: {context_id}")
        print(f"[DEBUG] Known contexts: {list(self.agents.keys())}")
        
        agent = self.agents.get(context_id)
        if not agent:
            print(f"[DEBUG] Creating NEW agent for context: {context_id}")
            agent = RAGAssessorAgent()
            self.agents[context_id] = agent
        else:
            print(f"[DEBUG] FOUND existing agent for context: {context_id}")
            print(f"[DEBUG] Agent's current_agent_name: {agent.current_agent_name}")

        # Create task updater for reporting progress
        updater = TaskUpdater(event_queue, task.id, context_id)

        # Start work and delegate to agent
        await updater.start_work()
        try:
            # Agent processes the message and uses updater to report results
            await agent.run(msg, updater)
            
            # If agent didn't explicitly complete/fail, mark as complete
            if not updater._terminal_state_reached:
                await updater.complete()
                
        except Exception as e:
            # Log and report failure
            print(f"Task {task.id} failed with error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}",
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(
        self, 
        context: RequestContext, 
        event_queue: EventQueue
    ) -> None:
        """
        Handle task cancellation request.
        
        Currently not supported - raises UnsupportedOperationError.
        
        Args:
            context: A2A request context
            event_queue: Queue for sending A2A events
        
        Raises:
            ServerError: Always raises UnsupportedOperationError
        """
        raise ServerError(error=UnsupportedOperationError())
    
    def cleanup_context(self, context_id: str) -> None:
        """
        Clean up resources for a completed context.
        
        Call this after an assessment is complete to free memory.
        
        Args:
            context_id: The context ID to clean up
        """
        if context_id in self.agents:
            del self.agents[context_id]
