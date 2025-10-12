"""Simple graph database store for batch nodes and edges."""

import os
import ssl
from typing import Any, Dict, List, Optional, Tuple

import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class GraphStore:
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        # Use provided values, then environment variables, then working defaults as fallback
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        print(f"GraphStore connecting to: {self.uri} with user: {self.username}")

        # Use the same SSL setup as the working example
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            ssl_context=ssl_ctx,
            connection_timeout=10,
            max_transaction_retry_time=5,
        )

        # Verify connectivity like the working example
        try:
            self.driver.verify_connectivity()
            print("GraphStore connected successfully!")
        except Exception as e:
            print(f"GraphStore connection failed: {e}")
            raise

    def close(self) -> None:
        self.driver.close()

    def add_nodes_batch(
        self, nodes: List[Dict[str, object]], label: str = "Node"
    ) -> None:
        with self.driver.session(database="neo4j") as session:
            query = f"""
                UNWIND $nodes AS node
                MERGE (n:{label} {{id: node.id}})
                SET n += node
            """
            session.run(query, nodes=nodes).consume()

    def add_edges_batch(
        self,
        edges: List[Tuple[str, str, str]],
        source_label: str = "Node",
        target_label: str = "Node",
    ) -> None:
        """Add edges in batch. Format: [(source_id, target_id, relationship_type)]"""
        with self.driver.session(database="neo4j") as session:
            # Group edges by relationship type and create separate queries
            edges_by_type: Dict[str, List[Dict[str, str]]] = {}
            for source, target, rel_type in edges:
                if rel_type not in edges_by_type:
                    edges_by_type[rel_type] = []
                edges_by_type[rel_type].append({"source": source, "target": target})

            # Create relationships for each type
            for rel_type, edge_list in edges_by_type.items():
                # Use a safe relationship name (replace special characters)
                safe_rel_type = rel_type.replace("-", "_").replace(" ", "_").upper()
                query = f"""
                    UNWIND $edges AS edge
                    MATCH (source:{source_label} {{id: edge.source}})
                    MATCH (target:{target_label} {{id: edge.target}})
                    MERGE (source)-[:{safe_rel_type}]->(target)
                """
                session.run(query, edges=edge_list)

    def get_document_store_component_ids(self, store_id: str) -> List[str]:
        """Fetch the component_node_ids from a DocumentStore node by exact ID."""
        with self.driver.session(database="neo4j") as session:
            query = """
                MATCH (d:DocumentStore {id: $store_id})
                RETURN d.component_node_ids AS component_ids
            """
            result = session.run(query, store_id=store_id).single()
            if result and result["component_ids"]:
                return list(result["component_ids"])
            return []

    def get_component_nodes_by_ids(
        self, component_ids: List[str]
    ) -> List[Dict[str, object]]:
        """Fetch multiple Component nodes by their IDs."""
        if not component_ids:
            return []

        with self.driver.session(database="neo4j") as session:
            query = """
                UNWIND $ids AS id
                MATCH (c:Component {id: id})
                RETURN c
            """
            results = session.run(query, ids=component_ids).data()
            return [dict(r["c"]) for r in results]

    def validate_user_exists(self, username: str) -> bool:
        """Check if a user exists in Neo4j."""
        with self.driver.session(database="neo4j") as session:
            query = """
                MATCH (u:User {username: $username})
                RETURN u.id AS user_id
            """
            result = session.run(query, username=username).single()
            return result is not None

    def get_pipeline_components_by_hash(
        self, pipeline_hash: str, username: str
    ) -> List[Dict[str, object]]:
        """
        Traverse entire pipeline graph using DFS to get all connected components.
        Only follows paths within the same pipeline.

        Args:
            pipeline_hash: Single pipeline name/hash to load
            username: Username to validate permissions

        Returns:
            List of component dictionaries with all necessary data
        """
        with self.driver.session(database="neo4j") as session:
            # First find the starting component(s) owned by the user for this pipeline
            start_query = """
                MATCH (u:User {username: $username})-[:OWNS]->(start:Component)
                WHERE start.pipeline_name = $pipeline_hash
                RETURN start.id AS start_id
            """
            start_results = session.run(
                query=start_query, pipeline_hash=pipeline_hash, username=username
            ).data()

            if not start_results:
                return []

            # Get all starting component IDs
            start_ids = [record["start_id"] for record in start_results]

            # Manual DFS traversal within the same pipeline
            return self._dfs_traversal_same_pipeline(session, start_ids, pipeline_hash)

    def _dfs_traversal_same_pipeline(
        self, session: Any, start_ids: List[str], pipeline_hash: str
    ) -> List[Dict[str, object]]:
        """DFS traversal that only follows components in the same pipeline."""
        visited = set()
        components = []
        stack = start_ids.copy()

        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue

            visited.add(current_id)

            # Get current node and ALL its connections (cross pipeline boundaries)
            query = """
                MATCH (c {id: $component_id})
                WHERE c:Component OR c:DocumentStore

                // Get ALL connections (don't filter by pipeline)
                OPTIONAL MATCH (c)-[:FLOWS_TO|READS_FROM|WRITES_TO]->(next)
                WHERE next:Component OR next:DocumentStore

                OPTIONAL MATCH (prev)-[:FLOWS_TO|READS_FROM|WRITES_TO]->(c)
                WHERE prev:Component OR prev:DocumentStore

                RETURN c,
                       collect(DISTINCT next.id) AS next_components,
                       collect(DISTINCT prev.id) AS prev_components,
                       labels(c) AS node_labels
            """

            result = session.run(
                query, component_id=current_id, pipeline_hash=pipeline_hash
            ).single()
            if result:
                component_data = dict(result["c"])
                next_components = result["next_components"]
                prev_components = result["prev_components"]
                node_labels = result["node_labels"]

                # Include ALL components (allows crossing pipeline boundaries)
                component_data["next_components"] = next_components
                component_data["prev_components"] = prev_components
                component_data["node_labels"] = node_labels
                components.append(component_data)

                # Only follow outgoing edges (next_components), not incoming (prev_components)
                # This prevents traversing backwards into other pipelines
                for next_id in next_components:
                    if next_id and next_id not in visited:
                        stack.append(next_id)

        return components
