from typing import Optional
from typing_extensions import Annotated
import typer
import streamlit as st

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from src.utils import initialize


def main(env_path: Annotated[Optional[str], typer.Argument()] = None):
    if env_path is not None:
        initialize(env_path)

    st.header('Simple Yuuri chatbot', divider='rainbow')

    with st.sidebar:
        top_k = st.slider(
            label="Top-k documents",
            max_value=4,
            min_value=1,
            step=1
        )
        use_hyde = st.checkbox("Use HyDE")

    storage_context = StorageContext.from_defaults(
        persist_dir="index/"
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.8)
        ]
    )

    if use_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = TransformQueryEngine(query_engine,
                                            query_transform=hyde)
    with st.chat_message("ai"):
        st.write("I can answer question related to Yuuri (優里) (Japanese singer)")
    with st.chat_message("user"):
        query = st.chat_input("Ask something here...")
        st.write(query)
    with st.chat_message("ai"):
        if query:
            response = query_engine.query(query)
            st.write(response.response)


if __name__ == '__main__':
    typer.run(main)