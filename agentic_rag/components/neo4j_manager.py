"""Simple graph database store for batch nodes and edges."""

import os
import ssl
from typing import Dict, List, Optional, Tuple

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
            print("✅ GraphStore connected successfully!")
        except Exception as e:
            print(f"❌ GraphStore connection failed: {e}")
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
