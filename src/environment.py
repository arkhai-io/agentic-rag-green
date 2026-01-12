"""RAG Environment for Purple Agents to interact with.

This module provides the RAG environment that Purple Agents use during assessments.
It wraps the agentic-rag SDK to provide:
- Agent registration (User nodes in Neo4j)
- Project management
- Pipeline creation and execution
- Document indexing
- Query execution with evaluation

Uses the existing agentic-rag SDK components:
- GraphStore: Neo4j operations
- PipelineFactory: Build pipelines from specs
- PipelineRunner: Execute pipelines
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agentic_rag import Config, PipelineFactory, set_global_config
from agentic_rag.components import GraphStore
from agentic_rag.pipeline import PipelineRunner
from dotenv import load_dotenv

from .models import (
    ActionResponse,
    ActionType,
    RegistrationResult,
    RegistrationStatus,
)

load_dotenv()


class RAGEnvironment:
    """
    The RAG environment that Purple Agents interact with.
    
    Wraps the agentic-rag SDK to provide a clean interface for:
    1. Agent registration (creates User nodes)
    2. Project management
    3. Pipeline creation and configuration
    4. Document indexing
    5. Query execution with evaluation
    
    Uses singleton pattern for efficient resource management.
    
    Example:
        env = RAGEnvironment()
        await env.initialize()
        
        # Register an agent
        result = await env.register_agent("my_agent")
        
        # Create a project
        await env.create_project("my_agent", "my_project")
        
        # Create a pipeline
        await env.create_pipeline(
            agent_name="my_agent",
            project_name="my_project",
            pipeline_name="my_pipeline",
            components=[{"type": "chunker", "name": "sentence_chunker"}, ...]
        )
    """

    _instance: Optional["RAGEnvironment"] = None
    _config: Optional[Config] = None
    _graph_store: Optional[GraphStore] = None
    _factory: Optional[PipelineFactory] = None
    _runner: Optional[PipelineRunner] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """
        Initialize the environment with agentic-rag SDK components.
        
        Reads configuration from environment variables:
        - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        - OPENROUTER_API_KEY (for LLM-based evaluators)
        
        Creates:
        - GraphStore: Neo4j connection for graph operations
        - PipelineFactory: For building pipelines from specs
        - PipelineRunner: For executing pipelines
        """
        if self._initialized:
            return

        # Create config from environment variables
        self._config = Config(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            lighthouse_api_key=os.getenv("LIGHTHOUSE_API_KEY"),
            log_level=os.getenv("AGENTIC_RAG_LOG_LEVEL", "INFO"),
        )

        if not self._config.validate_neo4j():
            raise ValueError(
                "Neo4j credentials required. Set environment variables:\n"
                "  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )

        # Set as global config for agentic-rag
        set_global_config(self._config)

        # Initialize SDK components
        self._graph_store = GraphStore(config=self._config)
        self._factory = PipelineFactory(config=self._config)
        self._runner = PipelineRunner(config=self._config, enable_caching=False)

        self._initialized = True
        print("RAGEnvironment: Initialized with agentic-rag SDK")

    async def close(self) -> None:
        """Close connections and clean up resources."""
        if self._graph_store:
            await self._graph_store.close_async()
        
        self._config = None
        self._graph_store = None
        self._factory = None
        self._runner = None
        self._initialized = False

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    async def register_agent(self, agent_name: str) -> RegistrationResult:
        """
        Register a new agent (Purple Agent) in the RAG environment.
        
        Creates a User node in Neo4j using the SDK's GraphStore.
        
        Args:
            agent_name: Unique identifier for the agent
        
        Returns:
            RegistrationResult with status and details
        """
        await self.initialize()

        # Validate agent name
        if not self._is_valid_agent_name(agent_name):
            return RegistrationResult(
                status=RegistrationStatus.INVALID_NAME,
                agent_name=agent_name,
                message="Invalid agent name. Must be 3-50 characters, alphanumeric with _ or -, start with letter.",
            )

        # Check if already exists
        if await self._graph_store.validate_user_exists_async(agent_name):
            return RegistrationResult(
                status=RegistrationStatus.ALREADY_EXISTS,
                agent_name=agent_name,
                message=f"Agent name '{agent_name}' is already taken.",
            )

        # Create user using SDK
        try:
            await self._graph_store.create_user_async(agent_name)
            
            return RegistrationResult(
                status=RegistrationStatus.SUCCESS,
                agent_name=agent_name,
                message=f"Agent '{agent_name}' registered successfully.",
                registered_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            return RegistrationResult(
                status=RegistrationStatus.ERROR,
                agent_name=agent_name,
                message=f"Registration failed: {str(e)}",
            )

    def _is_valid_agent_name(self, name: str) -> bool:
        """Validate agent name format."""
        import re
        if not name or len(name) < 3 or len(name) > 50:
            return False
        return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name))

    async def is_registered(self, agent_name: str) -> bool:
        """Check if an agent is registered."""
        await self.initialize()
        return await self._graph_store.validate_user_exists_async(agent_name)

    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================

    async def create_project(self, agent_name: str, project_name: str) -> ActionResponse:
        """
        Create a new project for an agent.
        
        Args:
            agent_name: The agent creating the project (must be registered)
            project_name: Unique name for the project
        
        Returns:
            ActionResponse with success status
        """
        await self.initialize()

        # Verify agent is registered
        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PROJECT,
                error=f"Agent '{agent_name}' is not registered.",
            )

        # Check if project exists
        if await self._graph_store.project_exists_async(agent_name, project_name):
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PROJECT,
                error=f"Project '{project_name}' already exists.",
            )

        # Create project using SDK
        try:
            await self._graph_store.create_project_async(agent_name, project_name)
            
            return ActionResponse(
                success=True,
                action=ActionType.CREATE_PROJECT,
                data={
                    "project_name": project_name,
                    "agent_name": agent_name,
                    "message": f"Project '{project_name}' created successfully.",
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PROJECT,
                error=f"Failed to create project: {str(e)}",
            )

    async def list_projects(self, agent_name: str) -> ActionResponse:
        """List all projects for an agent."""
        await self.initialize()

        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.LIST_PROJECTS,
                error=f"Agent '{agent_name}' is not registered.",
            )

        try:
            projects = await self._graph_store.get_user_projects_and_pipelines_async(agent_name)
            return ActionResponse(
                success=True,
                action=ActionType.LIST_PROJECTS,
                data={"projects": projects},
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.LIST_PROJECTS,
                error=f"Failed to list projects: {str(e)}",
            )

    # =========================================================================
    # PIPELINE MANAGEMENT
    # =========================================================================

    async def create_pipeline(
        self,
        agent_name: str,
        pipeline_name: str,
        project_name: str,
        components: List[Dict[str, Any]],
        pipeline_type: str = "indexing",
        config: Optional[Dict[str, Any]] = None,
    ) -> ActionResponse:
        """
        Create a new RAG pipeline using the SDK's PipelineFactory.
        
        Args:
            agent_name: The agent creating the pipeline
            pipeline_name: Unique name for the pipeline
            project_name: Project to create the pipeline in
            components: List of component specifications
            pipeline_type: Type of pipeline (indexing or retrieval)
            config: Optional pipeline configuration
        
        Returns:
            ActionResponse with creation status
        """
        await self.initialize()

        # Verify agent is registered
        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PIPELINE,
                error=f"Agent '{agent_name}' is not registered.",
            )

        # Verify project exists
        if not await self._graph_store.project_exists_async(agent_name, project_name):
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PIPELINE,
                error=f"Project '{project_name}' does not exist. Create it first.",
            )

        # Check if pipeline exists
        if await self._graph_store.pipeline_exists_async(agent_name, project_name, pipeline_name):
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PIPELINE,
                error=f"Pipeline '{pipeline_name}' already exists in project '{project_name}'.",
            )

        # Create pipeline using SDK's PipelineFactory
        try:
            if config is None:
                config = {}
            config["_pipeline_name"] = pipeline_name

            await self._factory.build_pipeline_graphs_from_specs_async(
                pipeline_specs=[components],
                username=agent_name,
                project=project_name,
                configs=[config],
                pipeline_types=[pipeline_type],
            )

            return ActionResponse(
                success=True,
                action=ActionType.CREATE_PIPELINE,
                data={
                    "pipeline_name": pipeline_name,
                    "project_name": project_name,
                    "pipeline_type": pipeline_type,
                    "components_count": len(components),
                    "message": f"Pipeline '{pipeline_name}' created successfully.",
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.CREATE_PIPELINE,
                error=f"Failed to create pipeline: {str(e)}",
            )

    async def list_pipelines(self, agent_name: str) -> ActionResponse:
        """List all pipelines for an agent across all projects."""
        await self.initialize()

        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.LIST_PIPELINES,
                error=f"Agent '{agent_name}' is not registered.",
            )

        try:
            projects = await self._graph_store.get_user_projects_and_pipelines_async(agent_name)
            return ActionResponse(
                success=True,
                action=ActionType.LIST_PIPELINES,
                data={"projects": projects},
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.LIST_PIPELINES,
                error=f"Failed to list pipelines: {str(e)}",
            )

    # =========================================================================
    # QUERY EXECUTION
    # =========================================================================

    async def query(
        self,
        agent_name: str,
        project_name: str,
        pipeline_name: str,
        query: str,
        ground_truth_answer: Optional[str] = None,
    ) -> ActionResponse:
        """
        Run a RAG query using the SDK's PipelineRunner.
        
        Args:
            agent_name: The agent running the query
            project_name: Project containing the pipeline
            pipeline_name: Name of the retrieval pipeline to use
            query: The question/query to answer
            ground_truth_answer: Optional ground truth for evaluation
        
        Returns:
            ActionResponse with query results
        """
        await self.initialize()

        # Verify agent is registered
        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.QUERY,
                error=f"Agent '{agent_name}' is not registered.",
            )

        try:
            # Load pipeline
            await self._runner.load_pipelines_async(
                pipeline_names=[pipeline_name],
                username=agent_name,
                project=project_name,
            )

            # Run query
            result = await self._runner.run_async(
                pipeline_name=pipeline_name,
                username=agent_name,
                type="retrieval",
                project=project_name,
                query=query,
                ground_truth_answer=ground_truth_answer,
            )

            return ActionResponse(
                success=True,
                action=ActionType.QUERY,
                data={
                    "query": query,
                    "result": result,
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.QUERY,
                error=f"Query failed: {str(e)}",
            )

    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================

    def _get_documents_path(self, agent_name: str, project_name: str) -> str:
        """Get the local path for storing documents."""
        root_dir = os.getenv("AGENTIC_ROOT_DIR", "./data")
        return os.path.join(root_dir, agent_name, project_name, "documents")

    async def upload_documents(
        self,
        agent_name: str,
        project_name: str,
        documents: List[Dict[str, Any]],
    ) -> ActionResponse:
        """
        Upload documents to local storage for later indexing.
        
        Args:
            agent_name: The agent uploading documents
            project_name: Project to upload documents to
            documents: List of dicts with filename, content_base64, content_type
        
        Returns:
            ActionResponse with upload results
        """
        import base64
        
        await self.initialize()

        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.UPLOAD_DOCUMENTS,
                error=f"Agent '{agent_name}' is not registered.",
            )

        # Verify project exists
        if not await self._graph_store.project_exists_async(agent_name, project_name):
            return ActionResponse(
                success=False,
                action=ActionType.UPLOAD_DOCUMENTS,
                error=f"Project '{project_name}' does not exist.",
            )

        try:
            # Create documents directory
            docs_path = self._get_documents_path(agent_name, project_name)
            os.makedirs(docs_path, exist_ok=True)

            uploaded_files = []
            total_size = 0

            for doc in documents:
                filename = doc.get("filename", "")
                content_b64 = doc.get("content_base64", "")
                
                if not filename or not content_b64:
                    continue

                # Decode and write file
                content = base64.b64decode(content_b64)
                file_path = os.path.join(docs_path, filename)
                
                with open(file_path, "wb") as f:
                    f.write(content)
                
                uploaded_files.append(filename)
                total_size += len(content)

            return ActionResponse(
                success=True,
                action=ActionType.UPLOAD_DOCUMENTS,
                data={
                    "project_name": project_name,
                    "uploaded_files": uploaded_files,
                    "storage_path": docs_path,
                    "total_size_bytes": total_size,
                    "file_count": len(uploaded_files),
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.UPLOAD_DOCUMENTS,
                error=f"Upload failed: {str(e)}",
            )

    async def list_documents(
        self,
        agent_name: str,
        project_name: str,
    ) -> ActionResponse:
        """
        List uploaded documents in a project.
        
        Args:
            agent_name: The agent listing documents
            project_name: Project to list documents from
        
        Returns:
            ActionResponse with list of documents
        """
        await self.initialize()

        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.LIST_DOCUMENTS,
                error=f"Agent '{agent_name}' is not registered.",
            )

        try:
            docs_path = self._get_documents_path(agent_name, project_name)
            
            if not os.path.exists(docs_path):
                return ActionResponse(
                    success=True,
                    action=ActionType.LIST_DOCUMENTS,
                    data={"documents": [], "storage_path": docs_path},
                )

            documents = []
            for filename in os.listdir(docs_path):
                file_path = os.path.join(docs_path, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    documents.append({
                        "filename": filename,
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })

            return ActionResponse(
                success=True,
                action=ActionType.LIST_DOCUMENTS,
                data={
                    "documents": documents,
                    "storage_path": docs_path,
                    "total_files": len(documents),
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.LIST_DOCUMENTS,
                error=f"Failed to list documents: {str(e)}",
            )

    # =========================================================================
    # INDEXING
    # =========================================================================

    async def index_documents(
        self,
        agent_name: str,
        project_name: str,
        pipeline_name: str,
        file_patterns: Optional[List[str]] = None,
    ) -> ActionResponse:
        """
        Index uploaded documents using an indexing pipeline.
        
        Args:
            agent_name: The agent indexing documents
            project_name: Project containing the documents
            pipeline_name: Name of the indexing pipeline to use
            file_patterns: Optional glob patterns to filter files
        
        Returns:
            ActionResponse with indexing results
        """
        import glob as glob_module
        
        await self.initialize()

        if not await self.is_registered(agent_name):
            return ActionResponse(
                success=False,
                action=ActionType.INDEX_DOCUMENTS,
                error=f"Agent '{agent_name}' is not registered.",
            )

        # Check pipeline exists
        if not await self._graph_store.pipeline_exists_async(agent_name, project_name, pipeline_name):
            return ActionResponse(
                success=False,
                action=ActionType.INDEX_DOCUMENTS,
                error=f"Pipeline '{pipeline_name}' does not exist.",
            )

        try:
            docs_path = self._get_documents_path(agent_name, project_name)
            
            if not os.path.exists(docs_path):
                return ActionResponse(
                    success=False,
                    action=ActionType.INDEX_DOCUMENTS,
                    error=f"No documents found. Upload documents first.",
                )

            # Get files to index
            if file_patterns:
                files = []
                for pattern in file_patterns:
                    files.extend(glob_module.glob(os.path.join(docs_path, pattern)))
            else:
                files = [
                    os.path.join(docs_path, f) 
                    for f in os.listdir(docs_path) 
                    if os.path.isfile(os.path.join(docs_path, f))
                ]

            if not files:
                return ActionResponse(
                    success=False,
                    action=ActionType.INDEX_DOCUMENTS,
                    error="No files match the specified patterns.",
                )

            # Load and run indexing pipeline
            await self._runner.load_pipelines_async(
                pipeline_names=[pipeline_name],
                username=agent_name,
                project=project_name,
            )

            result = await self._runner.run_async(
                pipeline_name=pipeline_name,
                username=agent_name,
                type="indexing",
                project=project_name,
                data_path=docs_path,
            )

            return ActionResponse(
                success=True,
                action=ActionType.INDEX_DOCUMENTS,
                data={
                    "pipeline_name": pipeline_name,
                    "indexed_files": [os.path.basename(f) for f in files],
                    "file_count": len(files),
                    "result": result,
                },
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=ActionType.INDEX_DOCUMENTS,
                error=f"Indexing failed: {str(e)}",
            )

    # =========================================================================
    # ACTION DISPATCH
    # =========================================================================

    async def handle_action(
        self,
        action: ActionType,
        params: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> ActionResponse:
        """
        Dispatch an action request to the appropriate handler.
        
        Args:
            action: The type of action to perform
            params: Action-specific parameters
            agent_name: The agent making the request
        
        Returns:
            ActionResponse with the result
        """
        if action == ActionType.REGISTER:
            result = await self.register_agent(params.get("agent_name", ""))
            return ActionResponse(
                success=result.status == RegistrationStatus.SUCCESS,
                action=action,
                data={
                    "status": result.status.value,
                    "agent_name": result.agent_name,
                    "message": result.message,
                    "registered_at": str(result.registered_at) if result.registered_at else None,
                },
                error=None if result.status == RegistrationStatus.SUCCESS else result.message,
            )

        elif action == ActionType.CREATE_PROJECT:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.create_project(agent_name, params.get("project_name", "default"))

        elif action == ActionType.LIST_PROJECTS:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.list_projects(agent_name)

        elif action == ActionType.CREATE_PIPELINE:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.create_pipeline(
                agent_name=agent_name,
                pipeline_name=params.get("pipeline_name", ""),
                project_name=params.get("project_name", "default"),
                components=params.get("components", []),
                pipeline_type=params.get("pipeline_type", "indexing"),
                config=params.get("config"),
            )

        elif action == ActionType.LIST_PIPELINES:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.list_pipelines(agent_name)

        elif action == ActionType.QUERY:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.query(
                agent_name=agent_name,
                project_name=params.get("project_name", "default"),
                pipeline_name=params.get("pipeline_name", ""),
                query=params.get("query", ""),
                ground_truth_answer=params.get("ground_truth_answer"),
            )

        elif action == ActionType.UPLOAD_DOCUMENTS:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.upload_documents(
                agent_name=agent_name,
                project_name=params.get("project_name", "default"),
                documents=params.get("documents", []),
            )

        elif action == ActionType.LIST_DOCUMENTS:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.list_documents(
                agent_name=agent_name,
                project_name=params.get("project_name", "default"),
            )

        elif action == ActionType.INDEX_DOCUMENTS:
            if not agent_name:
                return ActionResponse(success=False, action=action, error="agent_name required")
            return await self.index_documents(
                agent_name=agent_name,
                project_name=params.get("project_name", "default"),
                pipeline_name=params.get("pipeline_name", ""),
                file_patterns=params.get("file_patterns"),
            )

        else:
            return ActionResponse(
                success=False,
                action=action,
                error=f"Action '{action}' is not implemented",
            )
