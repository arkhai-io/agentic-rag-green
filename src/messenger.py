"""A2A messaging utilities for Green Agent.

This module provides utilities for communicating with Purple Agents
via the A2A (Agent-to-Agent) protocol.

The A2A protocol enables standardized communication between agents:
- Messages are sent as JSON payloads
- Each conversation has a context_id for tracking
- Responses can be streamed or returned as a single message

Key Components:
--------------
- create_message(): Create an A2A protocol message
- send_message(): Send a message to another agent
- Messenger: Class for managing conversations with multiple agents
"""

import json
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, Consumer
from a2a.types import DataPart, Message, Part, Role, TextPart


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TIMEOUT = 7200  # 2 hours timeout for large assessments (100+ papers)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_message(
    *,
    role: Role = Role.user,
    text: str,
    context_id: Optional[str] = None,
) -> Message:
    """
    Create an A2A protocol message.
    
    This creates a properly formatted message that can be sent to another
    agent via the A2A protocol.
    
    Args:
        role: The role of the message sender (user or agent)
        text: The message content (can be plain text or JSON)
        context_id: Optional conversation context ID for continuing a conversation
    
    Returns:
        Message: A2A protocol message object
    
    Example:
        # Simple text message
        msg = create_message(text="Hello, agent!")
        
        # JSON action request
        msg = create_message(
            text=json.dumps({"action": "register", "params": {"agent_name": "my_agent"}}),
            context_id="abc123"
        )
    """
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    """
    Merge multiple message parts into a single string.
    
    A2A messages can have multiple parts (text, data, files, etc.).
    This function extracts and combines them into a single string.
    
    Args:
        parts: List of message parts
    
    Returns:
        str: Combined content from all parts
    """
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


# =============================================================================
# SEND MESSAGE FUNCTION
# =============================================================================

async def send_message(
    message: str,
    base_url: str,
    context_id: Optional[str] = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Optional[Consumer] = None,
) -> Dict[str, Any]:
    """
    Send a message to another agent via A2A protocol.
    
    This is the core function for agent-to-agent communication. It:
    1. Resolves the agent's card (metadata) from their URL
    2. Creates an A2A client
    3. Sends the message and collects the response
    
    Args:
        message: The message content to send (string or JSON string)
        base_url: The agent's A2A endpoint URL
        context_id: Optional context ID to continue a conversation
        streaming: Whether to stream the response
        timeout: Request timeout in seconds
        consumer: Optional event consumer for streaming
    
    Returns:
        Dict containing:
        - context_id: The conversation context ID
        - response: The agent's response text
        - status: Task status (if applicable)
    
    Example:
        # Send a registration request
        result = await send_message(
            message='{"action": "register", "params": {"agent_name": "my_agent"}}',
            base_url="http://green-agent:9009"
        )
        print(result["response"])  # The agent's response
    """
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        # Resolve agent card (agent's metadata/capabilities)
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        
        # Create A2A client
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        
        # Add event consumer if provided
        if consumer:
            await client.add_event_consumer(consumer)

        # Create and send message
        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs: Dict[str, Any] = {"response": "", "context_id": None}

        # Process events (if streaming=False, only one event is generated)
        async for event in client.send_message(outbound_msg):
            last_event = event

        # Extract response from the event
        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


# =============================================================================
# MESSENGER CLASS
# =============================================================================

class Messenger:
    """
    Manages conversations with multiple agents.
    
    The Messenger class maintains conversation context IDs for each agent,
    allowing for multi-turn conversations. It provides a simple interface
    for the Green Agent to communicate with Purple Agents.
    
    Attributes:
        _context_ids: Dict mapping agent URLs to their conversation context IDs
    
    Example:
        messenger = Messenger()
        
        # First message to an agent (new conversation)
        response = await messenger.talk_to_agent(
            message="Hello!",
            url="http://purple-agent:9010"
        )
        
        # Follow-up message (continues the conversation)
        response = await messenger.talk_to_agent(
            message="What's your status?",
            url="http://purple-agent:9010"
        )
        
        # Start fresh conversation
        response = await messenger.talk_to_agent(
            message="Let's start over",
            url="http://purple-agent:9010",
            new_conversation=True
        )
    """

    def __init__(self):
        """Initialize messenger with empty context tracking."""
        self._context_ids: Dict[str, str] = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """
        Send a message to another agent and get their response.
        
        This is the main method for agent communication. It handles:
        - Conversation context management
        - Sending the message via A2A protocol
        - Extracting the response
        
        Args:
            message: The message to send (string or JSON string)
            url: The agent's A2A endpoint URL
            new_conversation: If True, start a fresh conversation
            timeout: Request timeout in seconds
        
        Returns:
            str: The agent's response message
        
        Raises:
            RuntimeError: If the agent responds with a non-completed status
        
        Example:
            # Send an action request
            response = await messenger.talk_to_agent(
                message=json.dumps({
                    "action": "create_pipeline",
                    "params": {"pipeline_name": "my_pipeline", "components": [...]}
                }),
                url="http://green-agent:9009"
            )
        """
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            timeout=timeout,
        )
        
        # Check for non-completed status
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        
        # Update context ID for future messages
        self._context_ids[url] = outputs.get("context_id")
        
        return outputs["response"]

    def reset(self) -> None:
        """
        Reset all conversation contexts.
        
        Call this to clear all tracked conversations and start fresh
        with all agents.
        """
        self._context_ids = {}
    
    def reset_agent(self, url: str) -> None:
        """
        Reset conversation context for a specific agent.
        
        Args:
            url: The agent's URL to reset
        """
        if url in self._context_ids:
            del self._context_ids[url]
