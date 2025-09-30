import os
import logging
from dataclasses import dataclass
from typing import final
import configparser

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock, get_graph_db_lock
import pipmaster as pm

if not pm.is_installed("kuzu"):
    pm.install("kuzu")

import kuzu  # type: ignore

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set kuzu logger level to ERROR to suppress warning logs
logging.getLogger("kuzu").setLevel(logging.ERROR)


@final
@dataclass
class KuzuStorage(BaseGraphStorage):
    """Kuzu graph storage implementation for LightRAG.

    This class provides a Kuzu-based graph storage backend that supports
    all BaseGraphStorage operations using Cypher queries.
    """

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Read env and override the arg if present
        kuzu_workspace = os.environ.get("KUZU_WORKSPACE")
        if kuzu_workspace and kuzu_workspace.strip():
            workspace = kuzu_workspace

        # Default to 'base' when both arg and env are empty
        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

        # Initialize instance variables
        self._database = None
        self._connection = None
        self._initialized = False

        # Basic logging setup
        self.logger = logger

    def _get_workspace_label(self) -> str:
        """Return workspace label (guaranteed non-empty during initialization)"""
        return self.workspace

    async def _create_schema(self):
        """Create the database schema with Entity node table and DIRECTED relationship table"""
        try:
            # Create Entity node table
            create_entity_query = """
            CREATE NODE TABLE Entity(
                entity_id STRING PRIMARY KEY,
                entity_type STRING,
                description STRING,
                source_id STRING,
                content STRING
            )
            """

            # Execute the CREATE NODE TABLE query
            self._connection.execute(create_entity_query)
            self.logger.info(f"[{self.workspace}] Created Entity node table")

            # Create DIRECTED relationship table
            create_relationship_query = """
            CREATE REL TABLE DIRECTED(
                FROM Entity TO Entity,
                weight DOUBLE,
                source_id STRING,
                description STRING,
                keywords STRING
            )
            """

            # Execute the CREATE REL TABLE query
            self._connection.execute(create_relationship_query)
            self.logger.info(f"[{self.workspace}] Created DIRECTED relationship table")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Failed to create schema: {e}")
            raise

    async def initialize(self):
        """Initialize the Kuzu database connection and schema"""
        async with get_data_init_lock():
            # Get database path from environment variable
            db_path = os.environ.get("KUZU_DB_PATH")

            # Handle both file-based and in-memory database modes
            if db_path:
                # File-based storage with workspace isolation
                # Create namespace-based directory structure and database file
                workspace_dir = os.path.join(db_path, self.namespace)
                os.makedirs(workspace_dir, exist_ok=True)

                # Use workspace-specific database file: namespace/workspace.kuzu
                db_file = os.path.join(workspace_dir, f"{self.workspace}.kuzu")
                self.logger.info(f"[{self.workspace}] Using database file: {db_file}")

                self._database = kuzu.Database(db_file)
                self.logger.info(f"[{self.workspace}] Initialized file-based Kuzu database at: {db_file}")
            else:
                # In-memory storage (default) - no workspace isolation needed
                self._database = kuzu.Database()
                self.logger.info(f"[{self.workspace}] Initialized in-memory Kuzu database")

            # Create connection
            self._connection = kuzu.Connection(self._database)

            # Create schema
            await self._create_schema()

            self._initialized = True
            self.logger.info(f"[{self.workspace}] KuzuStorage initialized successfully")

    async def finalize(self):
        """Finalize the Kuzu storage and clean up resources"""
        # Placeholder implementation - will be completed in task 10
        pass

    async def index_done_callback(self) -> None:
        """Commit storage operations after indexing"""
        # Placeholder implementation
        pass

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources"""
        # Placeholder implementation - will be completed in task 9
        return {"status": "success", "message": "data dropped"}

    # Abstract methods from BaseGraphStorage - placeholder implementations

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """
        try:
            query = "MATCH (n:Entity {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
            result = self._connection.execute(query, {"entity_id": node_id})

            # Get the result - Kuzu QueryResult has has_next() and get_next() methods
            if result.has_next():
                row = result.get_next()
                return bool(row[0])
            return False

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        # Placeholder implementation - will be completed in task 4
        return False

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node."""
        # Placeholder implementation - will be completed in task 6
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge."""
        # Placeholder implementation - will be completed in task 6
        return 0

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """
        try:
            query = "MATCH (n:Entity {entity_id: $entity_id}) RETURN n.entity_id, n.entity_type, n.description, n.source_id, n.content"
            result = self._connection.execute(query, {"entity_id": node_id})

            if result.has_next():
                row = result.get_next()
                # Convert the row to a dictionary with proper property names
                return {
                    "entity_id": str(row[0]) if row[0] is not None else "",
                    "entity_type": str(row[1]) if row[1] is not None else "",
                    "description": str(row[2]) if row[2] is not None else "",
                    "source_id": str(row[3]) if row[3] is not None else "",
                    "content": str(row[4]) if row[4] is not None else "",
                }
            return None

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting node for {node_id}: {str(e)}")
            raise

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes."""
        # Placeholder implementation - will be completed in task 4
        return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node."""
        # Placeholder implementation - will be completed in task 6
        return None

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all nodes that are associated with the given chunk_ids."""
        # Placeholder implementation - will be completed in task 12
        return []

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all edges that are associated with the given chunk_ids."""
        # Placeholder implementation - will be completed in task 12
        return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """
        try:
            # Prepare the properties for insertion
            properties = dict(node_data)
            if "entity_id" not in properties:
                properties["entity_id"] = node_id

            # Use MERGE to create or update the node
            query = """
            MERGE (n:Entity {entity_id: $entity_id})
            SET n.entity_type = $entity_type,
                n.description = $description,
                n.source_id = $source_id,
                n.content = $content
            """

            self._connection.execute(query, properties)
            self.logger.debug(f"[{self.workspace}] Upserted node: {node_id}")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error during upsert for node {node_id}: {str(e)}")
            raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph."""
        # Placeholder implementation - will be completed in task 4
        pass

    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        # Placeholder implementation - will be completed in task 9
        pass

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes"""
        # Placeholder implementation - will be completed in task 9
        pass

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges"""
        # Placeholder implementation - will be completed in task 9
        pass

    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph."""
        # Placeholder implementation - will be completed in task 8
        return []

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph of nodes."""
        # Placeholder implementation - will be completed in task 7
        return KnowledgeGraph(nodes={}, edges={})

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph."""
        # Placeholder implementation - will be completed in task 8
        return []

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph."""
        # Placeholder implementation - will be completed in task 8
        return []

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree."""
        # Placeholder implementation - will be completed in task 8
        return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching."""
        # Placeholder implementation - will be completed in task 8
        return []
