"""Pydantic models for A2A request/response handling.

This module defines the data structures used for communication between
the Green Agent (assessor) and Purple Agents (participants) via A2A protocol.

Key Models:
-----------
- EvalRequest: Incoming assessment request from AgentBeats platform
- ActionRequest: Request from Purple Agent to perform an action (register, query, etc.)
- ActionResponse: Response to Purple Agent's action request
- RegistrationResult: Result of agent registration attempt
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


# =============================================================================
# ENUMS - Define the types of actions Purple Agent can request
# =============================================================================

class ActionType(str, Enum):
    """
    Types of actions a Purple Agent can request from the Green Agent.
    
    The Green Agent acts as the environment/sandbox that provides:
    - Registration: Purple agents register to use the RAG environment
    - Pipeline operations: Create, modify, delete pipelines
    - Document operations: Index, retrieve documents
    - Query operations: Run RAG queries
    """
    # Authentication/Registration
    REGISTER = "register"
    
    # Pipeline management
    CREATE_PIPELINE = "create_pipeline"
    MODIFY_PIPELINE = "modify_pipeline"
    DELETE_PIPELINE = "delete_pipeline"
    LIST_PIPELINES = "list_pipelines"
    
    # Project management
    CREATE_PROJECT = "create_project"
    LIST_PROJECTS = "list_projects"
    
    # Document operations
    UPLOAD_DOCUMENTS = "upload_documents"
    LIST_DOCUMENTS = "list_documents"
    INDEX_DOCUMENTS = "index_documents"
    
    # Query operations
    QUERY = "query"


class RegistrationStatus(str, Enum):
    """Status of a registration attempt."""
    SUCCESS = "success"
    ALREADY_EXISTS = "already_exists"
    INVALID_NAME = "invalid_name"
    ERROR = "error"


# =============================================================================
# A2A PROTOCOL MODELS - For AgentBeats platform communication
# =============================================================================

class EvalRequest(BaseModel):
    """
    Request format sent by the AgentBeats platform to green agents.
    
    This is the initial message the Green Agent receives when an assessment starts.
    It contains:
    - participants: Map of role names to A2A endpoint URLs
    - config: Assessment-specific configuration
    
    Example:
        {
            "participants": {
                "rag_agent": "http://purple-agent:9010"
            },
            "config": {
                "domain": "machine_learning",
                "num_tasks": 10,
                "metrics": ["rouge", "bleu", "answer_quality"]
            }
        }
    """
    participants: Dict[str, HttpUrl]  # role -> agent URL
    config: Dict[str, Any]


# =============================================================================
# ACTION REQUEST/RESPONSE - For Purple Agent -> Green Agent communication
# =============================================================================

class ActionRequest(BaseModel):
    """
    Request from Purple Agent to perform an action in the RAG environment.
    
    The Purple Agent sends these messages to interact with the Green Agent's
    RAG environment. The 'action' field determines what operation to perform,
    and 'params' contains action-specific parameters.
    
    Example (Registration):
        {
            "action": "register",
            "params": {
                "agent_name": "my_rag_agent"
            }
        }
    
    Example (Create Pipeline):
        {
            "action": "create_pipeline",
            "params": {
                "pipeline_name": "my_pipeline",
                "components": [
                    {"type": "chunker", "name": "sentence_chunker"},
                    {"type": "embedder", "name": "openai_embedder"}
                ]
            }
        }
    """
    action: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class ActionResponse(BaseModel):
    """
    Response from Green Agent to Purple Agent's action request.
    
    Contains:
    - success: Whether the action succeeded
    - action: The action that was performed
    - data: Action-specific response data
    - error: Error message if action failed
    
    Example (Success):
        {
            "success": true,
            "action": "register",
            "data": {
                "agent_name": "my_rag_agent",
                "status": "success",
                "message": "Agent registered successfully"
            }
        }
    
    Example (Failure):
        {
            "success": false,
            "action": "register",
            "error": "Agent name 'existing_agent' is already taken"
        }
    """
    success: bool
    action: ActionType
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# =============================================================================
# REGISTRATION MODELS - Specific to agent registration
# =============================================================================

class RegistrationRequest(BaseModel):
    """
    Request to register a new agent (Purple Agent) in the RAG environment.
    
    When a Purple Agent wants to use the Green Agent's RAG environment,
    it must first register with a unique agent name. This creates:
    1. A User node in Neo4j graph database
    2. An entry in the registered agents tracking
    
    Attributes:
        agent_name: Unique identifier for the agent (like a username)
    """
    agent_name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
        description="Unique agent name (3-50 chars, alphanumeric with _ or -, must start with letter)"
    )


class RegistrationResult(BaseModel):
    """
    Result of an agent registration attempt.
    
    Attributes:
        status: Registration outcome (success, already_exists, invalid_name, error)
        agent_name: The agent name that was requested
        message: Human-readable description of the result
        registered_at: Timestamp when registration occurred (if successful)
    """
    status: RegistrationStatus
    agent_name: str
    message: str
    registered_at: Optional[datetime] = None


# =============================================================================
# PIPELINE MODELS - For pipeline management operations
# =============================================================================

class ComponentSpec(BaseModel):
    """
    Specification for a pipeline component.
    
    Uses the CATEGORY.TYPE format from agentic-rag library:
    
    Available component types:
    - CONVERTER: TEXT, PDF, DOCX, MARKDOWN, HTML, MARKER_PDF, MARKITDOWN_PDF
    - CHUNKER: DOCUMENT_SPLITTER, MARKDOWN_AWARE, SEMANTIC
    - EMBEDDER: SENTENCE_TRANSFORMERS, SENTENCE_TRANSFORMERS_DOC
    - RETRIEVER: CHROMA_EMBEDDING, QDRANT_EMBEDDING
    - RANKER: SENTENCE_TRANSFORMERS_SIMILARITY
    - GENERATOR: PROMPT_BUILDER, OPENAI, OPENROUTER
    - WRITER: CHROMA_DOCUMENT_WRITER, QDRANT_DOCUMENT_WRITER
    - DOCUMENT_STORE: CHROMA, QDRANT
    - EVALUATOR: BLEU, ROUGE, METEOR, ANSWER_QUALITY, FACT_MATCHING, etc.
    
    Example:
        {"type": "CHUNKER.DOCUMENT_SPLITTER"}
        {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"}
        {"type": "WRITER.CHROMA_DOCUMENT_WRITER"}
    """
    type: str = Field(
        ..., 
        description="Component type in CATEGORY.TYPE format (e.g., 'CHUNKER.DOCUMENT_SPLITTER')"
    )


class CreatePipelineRequest(BaseModel):
    """
    Request to create a new RAG pipeline.
    
    Attributes:
        pipeline_name: Unique name for the pipeline within the project
        project_name: Project to create the pipeline in (defaults to "default")
        components: List of component specifications (CATEGORY.TYPE format)
        pipeline_type: Type of pipeline (indexing or retrieval)
        config: Optional pipeline-level configuration (component configs by name)
    
    Example:
        {
            "pipeline_name": "my_indexing_pipeline",
            "project_name": "ml_docs",
            "pipeline_type": "indexing",
            "components": [
                {"type": "CONVERTER.TEXT"},
                {"type": "CHUNKER.DOCUMENT_SPLITTER"},
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
                {"type": "WRITER.CHROMA_DOCUMENT_WRITER"}
            ],
            "config": {
                "chunker": {"split_length": 200, "split_overlap": 20},
                "document_embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            }
        }
    """
    pipeline_name: str
    project_name: str = "default"
    components: List[ComponentSpec]
    pipeline_type: str = "indexing"
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pipeline-level configuration (keyed by component registry name)"
    )


class CreateProjectRequest(BaseModel):
    """
    Request to create a new project.
    
    Attributes:
        project_name: Unique name for the project
    """
    project_name: str


# =============================================================================
# QUERY MODELS - For RAG query operations
# =============================================================================

class QueryRequest(BaseModel):
    """
    Request to run a RAG query.
    
    Attributes:
        pipeline_name: Name of the retrieval pipeline to use
        project_name: Project containing the pipeline
        query: The question/query to answer
        ground_truth_answer: Optional ground truth for evaluation
    """
    pipeline_name: str
    project_name: str = "default"
    query: str
    ground_truth_answer: Optional[str] = None


class QueryResult(BaseModel):
    """
    Result of a RAG query.
    
    Attributes:
        query: The original query
        answer: Generated answer
        retrieved_documents: Documents retrieved for context
        eval_metrics: Evaluation metrics (if ground truth provided)
    """
    query: str
    answer: str
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    eval_metrics: Optional[Dict[str, Any]] = None


# =============================================================================
# DOCUMENT MODELS - For document upload and management
# =============================================================================

class DocumentUpload(BaseModel):
    """
    A single document to upload.
    
    Attributes:
        filename: Name of the file (e.g., "paper.pdf", "notes.txt")
        content_base64: Base64-encoded file content
        content_type: MIME type (e.g., "application/pdf", "text/plain")
    """
    filename: str
    content_base64: str
    content_type: Optional[str] = None


class UploadDocumentsRequest(BaseModel):
    """
    Request to upload documents to a project.
    
    Documents are stored locally and can be indexed later with index_documents.
    
    Attributes:
        project_name: Project to upload documents to
        documents: List of documents to upload (base64 encoded)
    
    Example:
        {
            "action": "upload_documents",
            "params": {
                "project_name": "ml_docs",
                "documents": [
                    {
                        "filename": "paper.pdf",
                        "content_base64": "JVBERi0xLjQK...",
                        "content_type": "application/pdf"
                    }
                ]
            }
        }
    """
    project_name: str
    documents: List[DocumentUpload]


class UploadDocumentsResult(BaseModel):
    """
    Result of document upload operation.
    
    Attributes:
        project_name: Project documents were uploaded to
        uploaded_files: List of successfully uploaded filenames
        storage_path: Path where documents are stored
        total_size_bytes: Total size of uploaded documents
    """
    project_name: str
    uploaded_files: List[str]
    storage_path: str
    total_size_bytes: int


class IndexDocumentsRequest(BaseModel):
    """
    Request to index uploaded documents using a pipeline.
    
    Attributes:
        project_name: Project containing the documents
        pipeline_name: Indexing pipeline to use
        file_patterns: Optional glob patterns to filter files (default: all)
    
    Example:
        {
            "action": "index_documents",
            "params": {
                "project_name": "ml_docs",
                "pipeline_name": "indexer",
                "file_patterns": ["*.pdf", "*.txt"]
            }
        }
    """
    project_name: str
    pipeline_name: str
    file_patterns: Optional[List[str]] = None


class ListDocumentsRequest(BaseModel):
    """
    Request to list uploaded documents in a project.
    
    Attributes:
        project_name: Project to list documents from
    """
    project_name: str
