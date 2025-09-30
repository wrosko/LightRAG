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

    async def initialize(self):
        """Initialize the Kuzu database connection and schema"""
        # Placeholder implementation - will be completed in task 2
        pass

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
        """Check if a node exists in the graph."""
        # Placeholder implementation - will be completed in task 3
        return False

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
        """Get node by its ID, returning only node properties."""
        # Placeholder implementation - will be completed in task 3
        return None

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

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph."""
        # Placeholder implementation - will be completed in task 3
        pass

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
