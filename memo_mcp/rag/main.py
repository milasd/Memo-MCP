import argparse
import asyncio
import sys

from memo_mcp.config.settings import DATA_DIR, TOP_K
from memo_mcp.rag.config.rag_config import RAGConfig
from memo_mcp.rag.memo_rag import create_rag_system
from memo_mcp.utils.logging_setup import set_logger
from pathlib import Path


"""
Run RAG queries over memo journal entries.
"""


logger = set_logger()


async def run_query(
    query: str,
    top_k: int = TOP_K,
    data_dir: Path = DATA_DIR,
    vector_store: str = "chroma",
    rebuild: bool = False,
) -> None:
    """Run a RAG query on journal entries.

    Args:
        query: Search query text
        top_k: Number of results to return
        data_dir: Path to memo data directory
        vector_store: Vector store type (chroma, faiss, simple)
        rebuild: Force rebuild of the index
    """
    config = RAGConfig(
        vector_store_type=vector_store,
        data_root=data_dir,
        use_gpu=True,
        cache_embeddings=True,
    )

    rag = await create_rag_system(config, logger)

    try:
        # Build or check index
        if rebuild:
            logger.info("Rebuilding index...")
            await rag.build_index(force_rebuild=True)
        else:
            stats = await rag.get_stats()
            if stats["total_documents"] == 0:
                logger.info("Building index for the first time...")
                await rag.build_index()
            else:
                logger.info(f"Using existing index with {stats['total_documents']} documents")


        results = await rag.query(query, top_k=top_k)

        # Log query results summary
        if not results:
            logger.info(f"\nNo results found for: '{query}'")
        else:
            logger.info(f"\nFound {len(results)} results for: '{query}'\n")
            log_results(results)

    finally:
        await rag.close()


def log_results(results: list[dict]) -> None:
    """Log search results to console.

    Args:
        results: List of search result dictionaries
    """
    logger.info("=" * 80)

    for i, result in enumerate(results, 1):
        similarity_score = result.get("similarity_score", 0.0)
        metadata = result["metadata"]

        # Show preview of text (first 300 characters).
        text_preview = result["text"][:300]
        if len(result["text"]) > 300:
            text_preview += "..."

        logger.info(f"\n{i}. {metadata.file_name} ({metadata.date_created})")
        logger.info(f"   Similarity Score: {similarity_score:.3f}")
        logger.info(f"   Path: {metadata.file_path}")
        logger.info(f"\n   {text_preview}\n")
        logger.info("-" * 80)


def set_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the rag journal.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Query memo journal entries using RAG (Retrieval-Augmented Generation)",
        epilog="""
Examples:
  %(prog)s "how did I feel about work this year?"
  %(prog)s "last time I started a hobby" -k 10
  %(prog)s "my thoughts on AI" --rebuild --vector-store faiss
  %(prog)s "travel plans" -d /path/to/custom/data

The tool uses semantic search to find relevant journal entries based on your query.
Results are ranked by similarity score, and include file paths and content previews.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural language search query (e.g., 'how did I feel about work this year?')"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of top results to return (default: {TOP_K})"
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=DATA_DIR,
        metavar="PATH",
        help=f"Path to memo data directory with YYYY/MM/DD.md structure (default: {DATA_DIR})"
    )
    parser.add_argument(
        "-v", "--vector-store",
        type=str,
        choices=["chroma", "faiss", "simple"],
        default="chroma",
        help="Vector store backend: chroma (persistent), faiss (fast), simple (in-memory) (default: chroma)"
    )
    parser.add_argument(
        "-r", "--rebuild",
        action="store_true",
        help="Force rebuild of the search index (use after adding new entries)"
    )
    return parser


def main() -> None:
    parser = set_parser()
    args = parser.parse_args()

    try:
        asyncio.run(run_query(
            query=args.query,
            top_k=args.top_k,
            data_dir=args.data_dir,
            vector_store=args.vector_store,
            rebuild=args.rebuild,
        ))
    except KeyboardInterrupt:
        logger.error("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
