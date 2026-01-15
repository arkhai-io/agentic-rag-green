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
import os
import time
from typing import Any, Dict, List, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from pydantic import ValidationError

from .environment import RAGEnvironment
from .messenger import Messenger
from .models import ActionRequest, ActionResponse, ActionType, EvalRequest

# Import evaluators from agentic-rag
from agentic_rag.components.evaluators.bleu_evaluator import BLEUEvaluator
from agentic_rag.components.evaluators.rouge_evaluator import ROUGEEvaluator
from agentic_rag.components.evaluators.coherence_evaluator import CoherenceEvaluator


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
    required_config_keys: list[str] = [
        "agent_name",           # Purple agent's registered name
        "project_name",         # Purple agent's project name
        "indexing_pipeline",    # Purple agent's indexing pipeline name
        "retrieval_pipeline",   # Purple agent's retrieval pipeline name
    ]

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
        
        Assessment Flow:
        ----------------
        1. Purple agent provides their pipeline names via config
        2. Green agent loads benchmark documents and questions
        3. Green agent indexes documents using purple agent's indexing pipeline
        4. Green agent runs queries using purple agent's retrieval pipeline
        5. Green agent collects and returns results
        
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
            
            # Extract config values
            config = eval_request.config
            agent_name = config["agent_name"]
            project_name = config["project_name"]
            indexing_pipeline = config["indexing_pipeline"]
            retrieval_pipeline = config["retrieval_pipeline"]
            
            # Optional config
            benchmark_domain = config.get("domain", "female_longevity")
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Starting assessment for agent '{agent_name}'...")
            )
            
            # Initialize environment
            await self.environment.initialize()
            
            # =========================================================
            # STEP 1: Load benchmark data (documents + questions)
            # =========================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading benchmark data for domain: {benchmark_domain}")
            )
            
            benchmark = await self._load_benchmark(benchmark_domain)
            if not benchmark:
                await updater.reject(
                    new_agent_text_message(f"Failed to load benchmark for domain: {benchmark_domain}")
                )
                return
            
            documents = benchmark.get("documents", [])
            questions = benchmark.get("questions", [])
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Loaded {len(documents)} documents and {len(questions)} questions"
                )
            )
            
            # =========================================================
            # STEP 2: Upload benchmark documents to purple agent's project
            # =========================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Uploading documents to project '{project_name}'...")
            )
            
            upload_result = await self.environment.upload_documents(
                agent_name=agent_name,
                project_name=project_name,
                documents=documents,
            )
            
            if not upload_result.success:
                await updater.reject(
                    new_agent_text_message(f"Failed to upload documents: {upload_result.error}")
                )
                return
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Uploaded {upload_result.data.get('file_count', 0)} documents"
                )
            )
            
            # =========================================================
            # STEP 3: Index documents using purple agent's indexing pipeline
            # =========================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Indexing documents with pipeline '{indexing_pipeline}'..."
                )
            )
            
            start_time = time.time()
            
            index_result = await self.environment.index_documents(
                agent_name=agent_name,
                project_name=project_name,
                pipeline_name=indexing_pipeline,
            )
            
            indexing_time = time.time() - start_time
            
            if not index_result.success:
                await updater.reject(
                    new_agent_text_message(f"Failed to index documents: {index_result.error}")
                )
                return
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Indexed {index_result.data.get('file_count', 0)} documents in {indexing_time:.2f}s"
                )
            )
            
            # =========================================================
            # STEP 4: Run queries using purple agent's retrieval pipeline
            # =========================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Running {len(questions)} queries with pipeline '{retrieval_pipeline}'..."
                )
            )
            
            query_results: List[Dict[str, Any]] = []
            query_start_time = time.time()
            
            for i, qa in enumerate(questions):
                question = qa.get("question", "")
                question_id = qa.get("id", f"q{i+1}")
                ground_truth = qa.get("ground_truth", None)
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running query {i+1}/{len(questions)}: {question[:50]}...")
                )
                
                # Run query through purple agent's retrieval pipeline
                query_result = await self.environment.query(
                    agent_name=agent_name,
                    project_name=project_name,
                    pipeline_name=retrieval_pipeline,
                    query=question,
                )
                
                # Extract only relevant data (remove raw embeddings to keep response small)
                answer_data = None
                answer_text = ""
                if query_result.success and query_result.data:
                    result = query_result.data.get("result", {})
                    # Keep only documents with content and scores, not raw embeddings
                    clean_docs = []
                    for doc in result.get("documents", []):
                        doc_content = doc.get("content", "")
                        clean_docs.append({
                            "content": doc_content[:500],  # Truncate content
                            "score": doc.get("score"),
                            "file_path": doc.get("meta", {}).get("file_path"),
                        })
                        # Use first document content as the answer
                        if not answer_text and doc_content:
                            answer_text = doc_content
                    answer_data = {
                        "total_documents": result.get("total_documents", 0),
                        "documents": clean_docs,
                    }
                
                # Run evaluation on the answer
                evaluation_scores = {}
                if query_result.success and answer_text:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Evaluating answer {i+1}/{len(questions)}...")
                    )
                    evaluation_scores = self._evaluate_answer(
                        question=question,
                        answer=answer_text,
                        ground_truth=ground_truth,
                    )
                
                query_results.append({
                    "question_id": question_id,
                    "question": question,
                    "ground_truth": ground_truth[:200] if ground_truth else None,
                    "success": query_result.success,
                    "answer": answer_data,
                    "evaluation": evaluation_scores,
                    "error": query_result.error if not query_result.success else None,
                })
            
            total_query_time = time.time() - query_start_time
            total_time = time.time() - start_time
            
            # =========================================================
            # STEP 5: Compile results and aggregate scores
            # =========================================================
            successful_queries = sum(1 for r in query_results if r["success"])
            aggregated_scores = self._aggregate_scores(query_results)
            
            assessment_results = {
                "agent_name": agent_name,
                "project_name": project_name,
                "indexing_pipeline": indexing_pipeline,
                "retrieval_pipeline": retrieval_pipeline,
                "benchmark_domain": benchmark_domain,
                "documents_indexed": index_result.data.get("file_count", 0),
                "questions_total": len(questions),
                "questions_answered": successful_queries,
                "indexing_time_seconds": round(indexing_time, 2),
                "query_time_seconds": round(total_query_time, 2),
                "total_time_seconds": round(total_time, 2),
                "evaluation_scores": aggregated_scores,
                "query_results": query_results,
            }
            
            # Build score summary string
            score_summary = f"{successful_queries}/{len(questions)} queries"
            if aggregated_scores.get("avg_rouge_l"):
                score_summary += f", ROUGE-L: {aggregated_scores['avg_rouge_l']:.3f}"
            if aggregated_scores.get("avg_bleu"):
                score_summary += f", BLEU: {aggregated_scores['avg_bleu']:.3f}"
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Assessment complete: {score_summary}")
            )
            
            # Return results as artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"Assessment completed for agent '{agent_name}'")),
                    Part(root=DataPart(data={
                        "participants": {
                            "rag_agent": str(eval_request.participants.get("rag_agent", agent_name))
                        },
                        "results": [{
                            "pass_rate": successful_queries / len(questions) if questions else 0,
                            "time_used": round(total_time, 2),
                            "max_score": len(questions),
                            "questions_answered": successful_queries,
                            # Add evaluation scores to results for leaderboard
                            "avg_rouge_l": aggregated_scores.get("avg_rouge_l", 0),
                            "avg_bleu": aggregated_scores.get("avg_bleu", 0),
                            "avg_coherence": aggregated_scores.get("avg_coherence", 0),
                        }],
                        "details": assessment_results,
                    }))
                ],
                name="Assessment Results",
            )
            
        except ValidationError as e:
            await updater.reject(
                new_agent_text_message(f"Invalid EvalRequest: {e}")
            )
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Assessment failed with error: {str(e)}")
            )

    async def _load_benchmark(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Load benchmark data (documents + questions) for a given domain.
        
        Looks for benchmark files in data/benchmarks/<domain>/
        
        Args:
            domain: The benchmark domain (e.g., "female_longevity")
        
        Returns:
            Dict with "documents" and "questions" keys, or None if not found
        """
        import base64
        
        # Determine benchmark path
        root_dir = os.getenv("AGENTIC_ROOT_DIR", "./data")
        benchmark_dir = os.path.join(root_dir, "benchmarks", domain)
        
        if not os.path.exists(benchmark_dir):
            print(f"Benchmark directory not found: {benchmark_dir}")
            return None
        
        # Load documents from papers/ subdirectory
        documents = []
        papers_dir = os.path.join(benchmark_dir, "papers")
        
        if os.path.exists(papers_dir):
            for filename in os.listdir(papers_dir):
                file_path = os.path.join(papers_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "rb") as f:
                        content = f.read()
                    
                    documents.append({
                        "filename": filename,
                        "content_base64": base64.b64encode(content).decode("utf-8"),
                    })
        
        # Load questions from qa_pairs.json
        questions = []
        qa_file = os.path.join(benchmark_dir, "qa_pairs.json")
        
        if os.path.exists(qa_file):
            with open(qa_file, "r") as f:
                qa_data = json.load(f)
                questions = qa_data.get("questions", [])
        
        return {
            "documents": documents,
            "questions": questions,
        }

    def _evaluate_answer(
        self,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate an answer using multiple metrics.
        
        Runs:
        - BLEU (requires ground truth) - lexical overlap
        - ROUGE (requires ground truth) - recall-oriented overlap
        - Coherence (no ground truth needed) - semantic coherence
        
        Args:
            question: The original question
            answer: The generated/retrieved answer
            ground_truth: Optional expected answer for comparison
        
        Returns:
            Dict with evaluation scores
        """
        scores = {}
        
        try:
            # Initialize evaluators
            bleu_evaluator = BLEUEvaluator(max_n=4, smoothing=True)
            rouge_evaluator = ROUGEEvaluator(rouge_type="rougeL", use_stemmer=True)
            coherence_evaluator = CoherenceEvaluator()
            
            # Run BLEU evaluation (requires ground truth)
            if ground_truth:
                bleu_result = bleu_evaluator.run(
                    query=question,
                    replies=[answer],
                    ground_truth_answer=ground_truth,
                )
                bleu_metrics = bleu_result.get("eval_data", {}).get("eval_metrics", {})
                if "bleu_4" in bleu_metrics:
                    scores["bleu"] = bleu_metrics["bleu_4"]["score"]
            
            # Run ROUGE evaluation (requires ground truth)
            if ground_truth:
                rouge_result = rouge_evaluator.run(
                    query=question,
                    replies=[answer],
                    ground_truth_answer=ground_truth,
                )
                rouge_metrics = rouge_result.get("eval_data", {}).get("eval_metrics", {})
                # ROUGE key is lowercase "rougel" and uses "score" field
                if "rougel" in rouge_metrics:
                    scores["rouge_l"] = rouge_metrics["rougel"]["score"]
            
            # Run Coherence evaluation (no ground truth needed)
            coherence_result = coherence_evaluator.run(
                query=question,
                replies=[answer],
            )
            coherence_metrics = coherence_result.get("eval_data", {}).get("eval_metrics", {})
            if "coherence" in coherence_metrics:
                scores["coherence"] = coherence_metrics["coherence"]["score"]
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            scores["evaluation_error"] = str(e)
        
        return scores

    def _aggregate_scores(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation scores across all queries.
        
        Args:
            query_results: List of query results with evaluation scores
        
        Returns:
            Dict with aggregated scores (averages)
        """
        # Collect all scores by metric
        metric_scores: Dict[str, List[float]] = {
            "bleu": [],
            "rouge_l": [],
            "coherence": [],
        }
        
        for result in query_results:
            eval_scores = result.get("evaluation", {})
            for metric, value in eval_scores.items():
                if metric in metric_scores and isinstance(value, (int, float)):
                    metric_scores[metric].append(value)
        
        # Calculate averages
        aggregated = {}
        for metric, values in metric_scores.items():
            if values:
                aggregated[f"avg_{metric}"] = round(sum(values) / len(values), 4)
                aggregated[f"{metric}_count"] = len(values)
        
        return aggregated

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
