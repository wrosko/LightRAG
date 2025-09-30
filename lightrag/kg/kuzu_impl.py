import os
import logging
from dataclasses import dataclass
from typing import final
import configparser

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
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

    def _execute_query_with_retry(self, query: str, parameters: dict = None) -> any:
        """Execute a query with retry logic for transient failures

        Args:
            query: Cypher query string
            parameters: Query parameters dictionary

        Returns:
            QueryResult from Kuzu

        Raises:
            RuntimeError: If query execution fails after retries
        """
        if parameters is None:
            parameters = {}

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.5) + wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((RuntimeError, ConnectionError, OSError)),
        )
        def _execute():
            if not self._initialized or not self._connection:
                raise RuntimeError("KuzuStorage not properly initialized")
            return self._connection.execute(query, parameters)

        try:
            return _execute()
        except Exception as e:
            self.logger.error(f"[{self.workspace}] Query execution failed after retries: {query[:100]}...: {str(e)}")
            raise RuntimeError(f"Query execution failed: {str(e)}") from e

    async def _create_schema(self):
        """Create the database schema with Entity node table and DIRECTED relationship table"""
        try:
            # Try to create Entity node table (will fail silently if it already exists)
            try:
                create_entity_query = """
                CREATE NODE TABLE Entity(
                    entity_id STRING PRIMARY KEY,
                    entity_type STRING,
                    description STRING,
                    source_id STRING,
                    content STRING
                )
                """
                self._connection.execute(create_entity_query)
                self.logger.info(f"[{self.workspace}] Created Entity node table")
            except Exception as e:
                if "already exists" in str(e).lower() or "exists in catalog" in str(e).lower():
                    self.logger.info(f"[{self.workspace}] Entity node table already exists")
                else:
                    raise

            # Try to create DIRECTED relationship table (will fail silently if it already exists)
            try:
                create_relationship_query = """
                CREATE REL TABLE DIRECTED(
                    FROM Entity TO Entity,
                    weight DOUBLE,
                    source_id STRING,
                    description STRING,
                    keywords STRING
                )
                """
                self._connection.execute(create_relationship_query)
                self.logger.info(f"[{self.workspace}] Created DIRECTED relationship table")
            except Exception as e:
                if "already exists" in str(e).lower() or "exists in catalog" in str(e).lower():
                    self.logger.info(f"[{self.workspace}] DIRECTED relationship table already exists")
                else:
                    raise

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Failed to create schema: {e}")
            raise

    async def _cleanup_on_error(self):
        """Clean up resources when initialization fails"""
        try:
            if self._connection:
                try:
                    self._connection.close()
                except Exception:
                    pass  # Ignore errors during cleanup
                self._connection = None

            if self._database:
                self._database = None

            self._initialized = False
        except Exception:
            pass  # Ignore all errors during cleanup

    async def initialize(self):
        """Initialize the Kuzu database connection and schema"""
        async with get_data_init_lock():
            try:
                # Get database path from environment variable
                db_path = os.environ.get("KUZU_DB_PATH")

                # Validate database path if provided
                if db_path:
                    if not os.path.isabs(db_path):
                        raise ValueError(f"KUZU_DB_PATH must be an absolute path: {db_path}")

                    # Check write permissions
                    if os.path.exists(db_path) and not os.access(db_path, os.W_OK):
                        raise PermissionError(f"No write permission for KUZU_DB_PATH: {db_path}")
                    elif not os.path.exists(db_path) and not os.access(os.path.dirname(db_path) or '/', os.W_OK):
                        raise PermissionError(f"Cannot create directory for KUZU_DB_PATH: {db_path}")

                # Handle both file-based and in-memory database modes
                if db_path:
                    # File-based storage with workspace isolation
                    # Create namespace-based directory structure and database file
                    workspace_dir = os.path.join(db_path, self.namespace)
                    os.makedirs(workspace_dir, exist_ok=True)

                    # Use workspace-specific database file: namespace/workspace.kuzu
                    db_file = os.path.join(workspace_dir, f"{self.workspace}.kuzu")
                    self.logger.info(f"[{self.workspace}] Using database file: {db_file}")

                    try:
                        self._database = kuzu.Database(db_file)
                        self.logger.info(f"[{self.workspace}] Initialized file-based Kuzu database at: {db_file}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to initialize Kuzu database at {db_file}: {str(e)}")
                else:
                    # In-memory storage (default) - no workspace isolation needed
                    try:
                        self._database = kuzu.Database()
                        self.logger.info(f"[{self.workspace}] Initialized in-memory Kuzu database")
                    except Exception as e:
                        raise RuntimeError(f"Failed to initialize in-memory Kuzu database: {str(e)}")

                # Create connection with timeout handling
                try:
                    self._connection = kuzu.Connection(self._database)
                    self.logger.info(f"[{self.workspace}] Kuzu connection established")
                except Exception as e:
                    raise RuntimeError(f"Failed to establish Kuzu connection: {str(e)}")

                # Create schema
                await self._create_schema()

                self._initialized = True
                self.logger.info(f"[{self.workspace}] KuzuStorage initialized successfully")

            except Exception as e:
                # Clean up on initialization failure
                await self._cleanup_on_error()
                raise RuntimeError(f"KuzuStorage initialization failed: {str(e)}") from e

    async def finalize(self):
        """Finalize the Kuzu storage and clean up resources"""
        async with get_graph_db_lock():
            try:
                if self._connection:
                    # Close the connection
                    self._connection.close()
                    self._connection = None
                    self.logger.info(f"[{self.workspace}] Kuzu connection closed")

                if self._database:
                    # For in-memory databases, this effectively clears the data
                    # For file-based databases, the file persists but connection is closed
                    self._database = None
                    self.logger.info(f"[{self.workspace}] Kuzu database resources cleaned up")

                self._initialized = False

            except Exception as e:
                self.logger.error(f"[{self.workspace}] Error during Kuzu storage finalization: {str(e)}")
                raise

    async def __aenter__(self):
        """Enter the async context manager"""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure resources are cleaned up when context manager exits"""
        await self.finalize()

    async def index_done_callback(self) -> None:
        """Commit storage operations after indexing"""
        # Placeholder implementation
        pass

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This method deletes all nodes and relationships in the current workspace.
        For file-based storage, this effectively clears the database file.
        For in-memory storage, this resets the database to empty state.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            # Delete all relationships first (delete directed relationships)
            delete_edges_query = "MATCH ()-[r:DIRECTED]->() DELETE r"
            self._connection.execute(delete_edges_query)

            # Delete all nodes
            delete_nodes_query = "MATCH (n:Entity) DELETE n"
            self._connection.execute(delete_nodes_query)

            self.logger.info(f"[{self.workspace}] Dropped all data from Kuzu storage")
            return {"status": "success", "message": "data dropped"}

        except Exception as e:
            error_msg = f"Failed to drop data: {str(e)}"
            self.logger.error(f"[{self.workspace}] {error_msg}")
            return {"status": "error", "message": error_msg}

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
            result = self._execute_query_with_retry(query, {"entity_id": node_id})

            # Get the result - Kuzu QueryResult has has_next() and get_next() methods
            if result.has_next():
                row = result.get_next()
                return bool(row[0])
            return False

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """
        try:
            query = """
            MATCH (a:Entity {entity_id: $source_entity_id})-[r:DIRECTED]-(b:Entity {entity_id: $target_entity_id})
            RETURN count(r) > 0 AS edge_exists
            """
            result = self._connection.execute(query, {
                "source_entity_id": source_node_id,
                "target_entity_id": target_node_id
            })

            if result.has_next():
                row = result.get_next()
                return bool(row[0])
            return False

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}")
            raise

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node.

        Args:
            node_id: The ID of the node

        Returns:
            The number of edges connected to the node
        """
        try:
            query = """
            MATCH (a:Entity {entity_id: $node_id})-[r:DIRECTED]-(b:Entity)
            RETURN count(r) AS degree
            """
            result = self._connection.execute(query, {"node_id": node_id})

            if result.has_next():
                row = result.get_next()
                return int(row[0]) if row[0] is not None else 0
            return 0

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes).

        Args:
            src_id: The ID of the source node
            tgt_id: The ID of the target node

        Returns:
            The sum of the degrees of the source and target nodes
        """
        try:
            src_degree = await self.node_degree(src_id)
            tgt_degree = await self.node_degree(tgt_id)
            return src_degree + tgt_degree

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting edge degree for {src_id} -> {tgt_id}: {str(e)}")
            raise

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Edge degrees as a batch using node_degrees_batch for bulk degree calculations.

        Args:
            edge_pairs: List of (source_id, target_id) tuples

        Returns:
            A dictionary mapping edge pairs to their total degrees
        """
        try:
            # Get all unique node IDs from the edge pairs
            node_ids = set()
            for src_id, tgt_id in edge_pairs:
                node_ids.add(src_id)
                node_ids.add(tgt_id)

            # Get degrees for all nodes in batch
            node_degrees = await self.node_degrees_batch(list(node_ids))

            # Calculate edge degrees
            edge_degrees = {}
            for src_id, tgt_id in edge_pairs:
                src_degree = node_degrees.get(src_id, 0)
                tgt_degree = node_degrees.get(tgt_id, 0)
                edge_degrees[(src_id, tgt_id)] = src_degree + tgt_degree

            return edge_degrees

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error in edge_degrees_batch: {str(e)}")
            raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """
        try:
            query = "MATCH (n:Entity {entity_id: $entity_id}) RETURN n.entity_id, n.entity_type, n.description, n.source_id, n.content"
            result = self._execute_query_with_retry(query, {"entity_id": node_id})

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
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """
        try:
            query = """
            MATCH (a:Entity {entity_id: $source_entity_id})-[r:DIRECTED]-(b:Entity {entity_id: $target_entity_id})
            RETURN r.weight, r.source_id, r.description, r.keywords
            """
            result = self._connection.execute(query, {
                "source_entity_id": source_node_id,
                "target_entity_id": target_node_id
            })

            if result.has_next():
                row = result.get_next()
                # Convert the row to a dictionary with proper property names and defaults
                return {
                    "weight": str(row[0]) if row[0] is not None else "1.0",
                    "source_id": str(row[1]) if row[1] is not None else "",
                    "description": str(row[2]) if row[2] is not None else "",
                    "keywords": str(row[3]) if row[3] is not None else "",
                }
            return None

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting edge between {source_node_id} and {target_node_id}: {str(e)}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes as a batch using UNWIND for efficient bulk retrieval.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        try:
            query = """
            UNWIND $node_ids AS id
            MATCH (n:Entity {entity_id: id})
            RETURN n.entity_id AS entity_id,
                   n.entity_type AS entity_type,
                   n.description AS description,
                   n.source_id AS source_id,
                   n.content AS content
            """
            result = self._connection.execute(query, {"node_ids": node_ids})
            nodes = {}

            # Process all results
            while result.has_next():
                row = result.get_next()
                entity_id = str(row[0]) if row[0] is not None else ""

                if entity_id:
                    nodes[entity_id] = {
                        "entity_id": entity_id,
                        "entity_type": str(row[1]) if row[1] is not None else "",
                        "description": str(row[2]) if row[2] is not None else "",
                        "source_id": str(row[3]) if row[3] is not None else "",
                        "content": str(row[4]) if row[4] is not None else "",
                    }

            return nodes

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error in get_nodes_batch: {str(e)}")
            raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get edges as a batch using UNWIND for bulk edge property retrieval.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        try:
            query = """
            UNWIND $pairs AS pair
            MATCH (a:Entity {entity_id: pair.src})-[r:DIRECTED]-(b:Entity {entity_id: pair.tgt})
            RETURN pair.src AS src_id, pair.tgt AS tgt_id,
                   r.weight AS weight, r.source_id AS source_id,
                   r.description AS description, r.keywords AS keywords
            """
            result = self._connection.execute(query, {"pairs": pairs})
            edges_dict = {}

            # Process all results
            while result.has_next():
                row = result.get_next()
                src_id = str(row[0]) if row[0] is not None else ""
                tgt_id = str(row[1]) if row[1] is not None else ""

                if src_id and tgt_id:
                    edge_props = {
                        "weight": str(row[2]) if row[2] is not None else "1.0",
                        "source_id": str(row[3]) if row[3] is not None else "",
                        "description": str(row[4]) if row[4] is not None else "",
                        "keywords": str(row[5]) if row[5] is not None else "",
                    }
                    edges_dict[(src_id, tgt_id)] = edge_props

            # For pairs that didn't have edges, add default properties
            for pair in pairs:
                src_id = pair["src"]
                tgt_id = pair["tgt"]
                if (src_id, tgt_id) not in edges_dict:
                    edges_dict[(src_id, tgt_id)] = {
                        "weight": "1.0",
                        "source_id": "",
                        "description": "",
                        "keywords": "",
                    }

            return edges_dict

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error in get_edges_batch: {str(e)}")
            raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """
        try:
            # First check if the node exists
            if not await self.has_node(source_node_id):
                return None

            query = """
            MATCH (a:Entity {entity_id: $node_id})-[r:DIRECTED]-(b:Entity)
            RETURN a.entity_id AS source_id, b.entity_id AS target_id
            """
            result = self._connection.execute(query, {"node_id": source_node_id})
            edges = []

            # Process all results
            while result.has_next():
                row = result.get_next()
                source_id = str(row[0]) if row[0] is not None else ""
                target_id = str(row[1]) if row[1] is not None else ""

                if source_id and target_id:
                    edges.append((source_id, target_id))

            return edges

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting node edges for {source_node_id}: {str(e)}")
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges as a batch using UNWIND for bulk edge retrieval.

        Args:
            node_ids: List of node IDs to get edges for

        Returns:
            A dictionary mapping each node_id to its list of (source_id, target_id) edge tuples
        """
        try:
            query = """
            UNWIND $node_ids AS node_id
            MATCH (a:Entity {entity_id: node_id})-[r:DIRECTED]-(b:Entity)
            RETURN node_id AS query_node_id, a.entity_id AS source_id, b.entity_id AS target_id
            """
            result = self._connection.execute(query, {"node_ids": node_ids})
            nodes_edges = {}

            # Initialize empty lists for all requested nodes
            for node_id in node_ids:
                nodes_edges[node_id] = []

            # Process all results
            while result.has_next():
                row = result.get_next()
                query_node_id = str(row[0]) if row[0] is not None else ""
                source_id = str(row[1]) if row[1] is not None else ""
                target_id = str(row[2]) if row[2] is not None else ""

                if query_node_id and source_id and target_id:
                    nodes_edges[query_node_id].append((source_id, target_id))

            return nodes_edges

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error in get_nodes_edges_batch: {str(e)}")
            raise

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all nodes that are associated with the given chunk_ids."""
        if not chunk_ids:
            return []

        # Build Cypher query to find nodes where source_id contains any of the chunk_ids
        # Since Kuzu may not support UNWIND like Neo4j, we'll use a different approach
        conditions = []
        parameters = {}

        for i, chunk_id in enumerate(chunk_ids):
            param_name = f"chunk_id_{i}"
            conditions.append(f"n.source_id IS NOT NULL AND n.source_id CONTAINS ${param_name}")
            parameters[param_name] = chunk_id

        where_clause = " OR ".join(conditions)

        query = f"""
        MATCH (n:Entity)
        WHERE {where_clause}
        RETURN DISTINCT n.entity_id AS entity_id, n.entity_type AS entity_type,
                        n.description AS description, n.source_id AS source_id, n.content AS content
        """

        result = self._connection.execute(query, parameters)
        nodes = []

        while result.has_next():
            row = result.get_next()
            node_dict = {
                "entity_id": str(row[0]) if row[0] is not None else "",
                "entity_type": str(row[1]) if row[1] is not None else "",
                "description": str(row[2]) if row[2] is not None else "",
                "source_id": str(row[3]) if row[3] is not None else "",
                "content": str(row[4]) if row[4] is not None else "",
                "id": str(row[0]) if row[0] is not None else "",  # Add id for compatibility
            }
            nodes.append(node_dict)

        result.close()
        return nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Get all edges that are associated with the given chunk_ids."""
        if not chunk_ids:
            return []

        # Build Cypher query to find edges where source_id contains any of the chunk_ids
        conditions = []
        parameters = {}

        for i, chunk_id in enumerate(chunk_ids):
            param_name = f"chunk_id_{i}"
            conditions.append(f"r.source_id IS NOT NULL AND r.source_id CONTAINS ${param_name}")
            parameters[param_name] = chunk_id

        where_clause = " OR ".join(conditions)

        query = f"""
        MATCH (a:Entity)-[r:DIRECTED]-(b:Entity)
        WHERE {where_clause}
        RETURN DISTINCT a.entity_id AS source, b.entity_id AS target,
                        r.weight AS weight, r.source_id AS source_id,
                        r.description AS description, r.keywords AS keywords
        """

        result = self._connection.execute(query, parameters)
        edges = []

        while result.has_next():
            row = result.get_next()
            edge_properties = {
                "source": str(row[0]) if row[0] is not None else "",
                "target": str(row[1]) if row[1] is not None else "",
                "weight": float(row[2]) if row[2] is not None else 1.0,
                "source_id": str(row[3]) if row[3] is not None else "",
                "description": str(row[4]) if row[4] is not None else "",
                "keywords": str(row[5]) if row[5] is not None else "",
            }
            edges.append(edge_properties)

        result.close()
        return edges

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, ConnectionError, OSError)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """
        try:
            # Validate that both source and target nodes exist
            source_exists = await self.has_node(source_node_id)
            target_exists = await self.has_node(target_node_id)

            if not source_exists:
                raise ValueError(f"Source node '{source_node_id}' does not exist")
            if not target_exists:
                raise ValueError(f"Target node '{target_node_id}' does not exist")

            # Prepare edge properties with defaults
            properties = dict(edge_data)

            # Ensure required properties exist with defaults
            if "weight" not in properties:
                properties["weight"] = 1.0
            if "source_id" not in properties:
                properties["source_id"] = ""
            if "description" not in properties:
                properties["description"] = ""
            if "keywords" not in properties:
                properties["keywords"] = ""

            # Convert weight to float for proper storage
            try:
                properties["weight"] = float(properties["weight"])
            except (ValueError, TypeError):
                properties["weight"] = 1.0

            # Use MERGE to create or update the directed edges (both directions for undirected graph)
            query = """
            MATCH (a:Entity {entity_id: $source_entity_id})
            MATCH (b:Entity {entity_id: $target_entity_id})
            MERGE (a)-[r1:DIRECTED]->(b)
            SET r1.weight = $weight,
                r1.source_id = $source_id,
                r1.description = $description,
                r1.keywords = $keywords
            MERGE (b)-[r2:DIRECTED]->(a)
            SET r2.weight = $weight,
                r2.source_id = $source_id,
                r2.description = $description,
                r2.keywords = $keywords
            """

            self._connection.execute(query, {
                "source_entity_id": source_node_id,
                "target_entity_id": target_node_id,
                "weight": properties["weight"],
                "source_id": properties["source_id"],
                "description": properties["description"],
                "keywords": properties["keywords"]
            })

            self.logger.debug(f"[{self.workspace}] Upserted edge: {source_node_id} -> {target_node_id}")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error during edge upsert between {source_node_id} and {target_node_id}: {str(e)}")
            raise

    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph.

        Args:
            node_id: The ID of the node to delete
        """
        try:
            query = """
            MATCH (n:Entity {entity_id: $entity_id})
            DETACH DELETE n
            """
            self._connection.execute(query, {"entity_id": node_id})
            self.logger.debug(f"[{self.workspace}] Deleted node: {node_id}")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error deleting node {node_id}: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        try:
            for node_id in nodes:
                await self.delete_node(node_id)
            self.logger.debug(f"[{self.workspace}] Deleted {len(nodes)} nodes")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error deleting nodes {nodes}: {str(e)}")
            raise

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        try:
            for source_id, target_id in edges:
                # Delete both directed edges since we store bidirectional relationships
                delete_directed_query = """
                MATCH (a:Entity {entity_id: $source_entity_id})-[r:DIRECTED]->(b:Entity {entity_id: $target_entity_id})
                DELETE r
                """
                self._connection.execute(delete_directed_query, {
                    "source_entity_id": source_id,
                    "target_entity_id": target_id
                })

                delete_reverse_query = """
                MATCH (a:Entity {entity_id: $target_entity_id})-[r:DIRECTED]->(b:Entity {entity_id: $source_entity_id})
                DELETE r
                """
                self._connection.execute(delete_reverse_query, {
                    "source_entity_id": source_id,
                    "target_entity_id": target_id
                })

                self.logger.debug(f"[{self.workspace}] Deleted edge: {source_id} <-> {target_id}")

            self.logger.debug(f"[{self.workspace}] Deleted {len(edges)} edges")

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error deleting edges {edges}: {str(e)}")
            raise

    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph.

        Returns:
            A list of all entity types (labels) in the graph, sorted alphabetically
        """
        try:
            query = """
            MATCH (n:Entity)
            RETURN DISTINCT n.entity_type AS entity_type
            """
            result = self._connection.execute(query)
            labels = []

            while result.has_next():
                row = result.get_next()
                entity_type = str(row[0]) if row[0] is not None else ""
                if entity_type:  # Only add non-empty labels
                    labels.append(entity_type)

            # Sort alphabetically in Python
            return sorted(labels)

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting all labels: {str(e)}")
            raise

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()

        try:
            if node_label == "*":
                # For all nodes, get nodes with highest degree first (up to max_nodes)
                query = """
                MATCH (n:Entity)
                OPTIONAL MATCH (n)-[r:DIRECTED]-(m:Entity)
                WITH n, count(r) AS degree
                ORDER BY degree DESC
                LIMIT $max_nodes
                RETURN n.entity_id AS entity_id
                """
                result_query = self._connection.execute(query, {"max_nodes": max_nodes})

                # Collect all node IDs
                node_ids = []
                while result_query.has_next():
                    row = result_query.get_next()
                    node_ids.append(str(row[0]))

                # Check if we have more nodes than requested
                if len(node_ids) >= max_nodes:
                    result.is_truncated = True
                    self.logger.info(f"[{self.workspace}] Graph truncated: at least {max_nodes} nodes found, limited to {max_nodes}")

                # Build the subgraph from these nodes
                await self._build_subgraph_from_nodes(node_ids, result, visited_nodes, visited_edges)

            else:
                # For specific starting node, do BFS traversal
                await self._bfs_subgraph_extraction(node_label, max_depth, max_nodes, result, visited_nodes, visited_edges)

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error in get_knowledge_graph for {node_label}: {str(e)}")
            raise

        return result

    async def _build_subgraph_from_nodes(self, node_ids: list[str], result: KnowledgeGraph,
                                        visited_nodes: set, visited_edges: set):
        """Build subgraph from a list of node IDs"""
        if not node_ids:
            return

        # Get node data for all nodes
        batch_nodes = await self.get_nodes_batch(node_ids)
        for node_id, node_data in batch_nodes.items():
            if node_id not in visited_nodes:
                visited_nodes.add(node_id)
                # Convert to KnowledgeGraphNode format
                kg_node = KnowledgeGraphNode(
                    id=node_id,
                    labels=[node_data.get("entity_type", "")],
                    properties=node_data
                )
                result.nodes.append(kg_node)

        # Get edges between these nodes
        for i, src_id in enumerate(node_ids):
            for tgt_id in node_ids[i+1:]:  # Avoid duplicate checks
                edge_data = await self.get_edge(src_id, tgt_id)
                if edge_data:
                    edge_key = (src_id, tgt_id)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        # Convert to KnowledgeGraphEdge format
                        kg_edge = KnowledgeGraphEdge(
                            id=f"{src_id}-{tgt_id}",
                            type="DIRECTED",  # Relationship type
                            source=src_id,
                            target=tgt_id,
                            properties=edge_data
                        )
                        result.edges.append(kg_edge)

    async def _bfs_subgraph_extraction(self, start_node_id: str, max_depth: int, max_nodes: int,
                                     result: KnowledgeGraph, visited_nodes: set, visited_edges: set):
        """Extract subgraph using BFS traversal from a starting node"""
        from collections import deque

        # Check if start node exists
        if not await self.has_node(start_node_id):
            self.logger.debug(f"[{self.workspace}] Start node {start_node_id} does not exist")
            return

        queue = deque([(start_node_id, 0)])  # (node_id, depth)
        node_count = 0

        while queue and node_count < max_nodes:
            current_node_id, depth = queue.popleft()

            if current_node_id in visited_nodes or depth > max_depth:
                continue

            # Add current node
            visited_nodes.add(current_node_id)
            node_count += 1

            # Get node data
            node_data = await self.get_node(current_node_id)
            if node_data:
                kg_node = KnowledgeGraphNode(
                    id=current_node_id,
                    labels=[node_data.get("entity_type", "")],
                    properties=node_data
                )
                result.nodes.append(kg_node)

            # Get connected edges and nodes
            node_edges = await self.get_node_edges(current_node_id)
            if node_edges:
                for src_id, tgt_id in node_edges:
                    # Determine the neighbor (not the current node)
                    neighbor_id = tgt_id if src_id == current_node_id else src_id

                    # Add edge if not already visited
                    edge_key = tuple(sorted([src_id, tgt_id]))  # Sort for consistent key
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)

                        # Get edge data
                        edge_data = await self.get_edge(src_id, tgt_id)
                        if edge_data:
                            kg_edge = KnowledgeGraphEdge(
                                id=f"{src_id}-{tgt_id}",
                                type="DIRECTED",  # Relationship type
                                source=src_id,
                                target=tgt_id,
                                properties=edge_data
                            )
                            result.edges.append(kg_edge)

                    # Add neighbor to queue if not visited and within depth limit
                    if neighbor_id not in visited_nodes and depth + 1 <= max_depth:
                        queue.append((neighbor_id, depth + 1))

        # Check if we truncated due to max_nodes
        if len(queue) > 0 or node_count >= max_nodes:
            result.is_truncated = True
            self.logger.info(f"[{self.workspace}] Graph truncated: reached max_nodes limit of {max_nodes}")

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        try:
            query = """
            MATCH (n:Entity)
            RETURN n.entity_id AS entity_id,
                   n.entity_type AS entity_type,
                   n.description AS description,
                   n.source_id AS source_id,
                   n.content AS content
            """
            result = self._connection.execute(query)
            nodes = []

            while result.has_next():
                row = result.get_next()
                node = {
                    "entity_id": str(row[0]) if row[0] is not None else "",
                    "entity_type": str(row[1]) if row[1] is not None else "",
                    "description": str(row[2]) if row[2] is not None else "",
                    "source_id": str(row[3]) if row[3] is not None else "",
                    "content": str(row[4]) if row[4] is not None else "",
                }
                nodes.append(node)

            return nodes

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting all nodes: {str(e)}")
            raise

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        try:
            query = """
            MATCH (a:Entity)-[r:DIRECTED]-(b:Entity)
            RETURN DISTINCT a.entity_id AS source, b.entity_id AS target,
                   r.weight AS weight, r.source_id AS source_id,
                   r.description AS description, r.keywords AS keywords
            """
            result = self._connection.execute(query)
            edges = []

            while result.has_next():
                row = result.get_next()
                edge = {
                    "source": str(row[0]) if row[0] is not None else "",
                    "target": str(row[1]) if row[1] is not None else "",
                    "weight": str(row[2]) if row[2] is not None else "1.0",
                    "source_id": str(row[3]) if row[3] is not None else "",
                    "description": str(row[4]) if row[4] is not None else "",
                    "keywords": str(row[5]) if row[5] is not None else "",
                }
                edges.append(edge)

            return edges

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting all edges: {str(e)}")
            raise

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities).

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of entity_ids sorted by degree (highest first)
        """
        try:
            query = f"""
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:DIRECTED]-(m:Entity)
            WITH n.entity_id AS entity_id, count(r) AS degree
            ORDER BY degree DESC
            LIMIT {limit}
            RETURN entity_id
            """
            result = self._connection.execute(query)
            labels = []

            while result.has_next():
                row = result.get_next()
                entity_id = str(row[0]) if row[0] is not None else ""
                if entity_id:
                    labels.append(entity_id)

            return labels

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            raise

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching entity_ids sorted by relevance
        """
        try:
            query_strip = query.strip()
            if not query_strip:
                return []

            # Use CONTAINS for simple text matching (case-sensitive)
            search_query = f"""
            MATCH (n:Entity)
            WHERE n.entity_id CONTAINS $query
            RETURN n.entity_id AS entity_id
            ORDER BY n.entity_id
            LIMIT {limit}
            """
            result = self._connection.execute(search_query, {"query": query_strip})
            labels = []

            while result.has_next():
                row = result.get_next()
                entity_id = str(row[0]) if row[0] is not None else ""
                if entity_id:
                    labels.append(entity_id)

            return labels

        except Exception as e:
            self.logger.error(f"[{self.workspace}] Error searching labels with query '{query}': {str(e)}")
            raise
