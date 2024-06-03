import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine, CompactAndRefine

from src.utils import initialize


# Perform question and answer using RAG system
def main(
        dotenv_path: str,
        query: str,
        persist_dir: str = "index",
        top_k: int = 4,
        use_hyde: bool = False
) -> None:
    initialize(dotenv_path)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.8)
        ]
    )
    if use_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = TransformQueryEngine(query_engine,
                                            query_transform=hyde)
    response = query_engine.query(query)
    print(response.response)


if __name__ == '__main__':
    typer.run(main)
