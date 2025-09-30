#!/usr/bin/env python3
"""
LightRAG with Ollama and KuzuStorage - Processing the Transformer Paper

This example demonstrates how to use LightRAG with Ollama (gemma3:1b)
and KuzuStorage to process and query the Transformer research paper.

Features:
- Uses Ollama gemma3:1b for both LLM and embeddings
- Uses KuzuStorage (embedded graph database) for knowledge graph storage
- Processes the complete Transformer research paper
- Demonstrates knowledge graph exploration and querying

Requirements:
- pip install lightrag-hku
- ollama run gemma3:1b

Usage:
    python lightrag_ollama_transformer_example.py
"""

import asyncio
import os
from pathlib import Path
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

# Configure working directory for storing the graph
WORKING_DIR = "./transformer_rag_storage"

# Optional: Configure Kuzu database path (uncomment to use file-based storage)
# os.environ["KUZU_DB_PATH"] = os.path.join(WORKING_DIR, "kuzu_db")


def setup_directories():
    """Create necessary directories for the example."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"📁 Working directory: {WORKING_DIR}")


async def configure_rag():
    """Configure LightRAG with Ollama backend."""

    # Initialize LightRAG with KuzuStorage
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma3:1b",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
            "timeout": 300,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,  # nomic-embed-text dimension
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434",
            ),
        ),
        # Explicitly use KuzuStorage for the graph (embedded database)
        graph_storage="KuzuStorage",
        # Vector storage - using NanoVectorDB (default)
        # Doc status storage - using JsonDocStatusStorage (default)
    )

    # Initialize storages (required for LightRAG)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def load_transformer_paper(rag):
    """Load and process the Transformer paper."""
    print("\n📄 Loading Transformer paper...")

    # Read the transformer paper
    transformer_paper_path = "transformerpaper.txt"

    if not os.path.exists(transformer_paper_path):
        raise FileNotFoundError(f"Transformer paper not found at: {transformer_paper_path}")

    with open(transformer_paper_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()

    print(f"📏 Paper length: {len(paper_content)} characters")
    print(f"📊 Estimated tokens: ~{len(paper_content.split()) * 1.3:.0f}")

    # Insert the document into LightRAG
    print("\n🚀 Processing document with LightRAG...")

    # Split into smaller chunks for better processing
    chunk_size = 2000  # characters per chunk
    chunks = []

    # Split by paragraphs first, then by chunk size
    paragraphs = paper_content.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"📦 Split into {len(chunks)} chunks for processing")

    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"  Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
        try:
            await rag.ainsert(chunk)
        except Exception as e:
            print(f"  ⚠️  Error processing chunk {i}: {e}")
            continue

    print("✅ Document processing complete!")


async def demonstrate_queries(rag):
    """Demonstrate various queries on the processed Transformer paper."""
    print("\n🔍 Running demonstration queries...")

    queries = [
        "What is the main contribution of the Transformer paper?",
        "How does the Transformer architecture work?",
        "What are the key components of the Transformers library?",
        "Explain self-attention mechanism in Transformers",
        "What are the differences between BERT and GPT models?",
        "How does the Model Hub work in the Transformers library?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)

        try:
            response = await rag.aquery(query)
            print(response)
        except Exception as e:
            print(f"❌ Error querying: {e}")

        print()


async def explore_knowledge_graph(rag):
    """Explore the knowledge graph structure."""
    print("\n🕸️  Exploring Knowledge Graph...")

    try:
        # Get knowledge graph for "Transformer"
        kg = await rag.get_knowledge_graph("Transformer", max_depth=2, max_nodes=20)
        print(f"📊 Knowledge Graph for 'Transformer':")
        print(f"   Nodes: {len(kg.nodes)}")
        print(f"   Edges: {len(kg.edges)}")

        # Show some sample nodes and edges
        print("\n📋 Sample nodes:")
        for i, node in enumerate(kg.nodes[:5]):
            print(f"   {i+1}. {node.get('entity_id', 'Unknown')} ({node.get('entity_type', 'Unknown')})")

        print("\n🔗 Sample edges:")
        for i, edge in enumerate(kg.edges[:5]):
            source = edge.get('source', 'Unknown')
            target = edge.get('target', 'Unknown')
            relation = edge.get('description', 'Unknown')
            print(f"   {i+1}. {source} -> {target}: {relation[:50]}...")

    except Exception as e:
        print(f"❌ Error exploring knowledge graph: {e}")


async def main():
    """Main execution function."""
    print("🤖 LightRAG with Ollama + KuzuStorage - Transformer Paper Example")
    print("=" * 60)

    # Setup
    setup_directories()

    # Initialize shared storage for multiprocessing support
    initialize_share_data(workers=1)

    try:
        # Configure and initialize LightRAG
        print("\n⚙️  Configuring LightRAG with Ollama...")
        rag = await configure_rag()
        print("✅ LightRAG initialized successfully")

        # Load and process the paper
        await load_transformer_paper(rag)

        # Explore the knowledge graph
        await explore_knowledge_graph(rag)

        # Demonstrate queries
        await demonstrate_queries(rag)

        print("\n🎉 Example completed successfully!")
        print(f"📁 Graph data stored in: {WORKING_DIR}")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
