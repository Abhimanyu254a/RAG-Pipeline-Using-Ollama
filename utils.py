"""
RAG Utility Functions — Modern LlamaIndex v0.10+ API
=====================================================

Helper functions for building and evaluating advanced RAG pipelines:

  Index builders
  ──────────────
  build_sentence_window_index  — VectorStoreIndex using Sentence-Window RAG
  build_automerging_index      — VectorStoreIndex using Auto-Merging RAG

  Query-engine factories
  ──────────────────────
  get_sentence_window_query_engine — query engine with metadata replacement + reranking
  get_automerging_query_engine     — query engine with auto-merge retrieval + reranking

  Evaluation
  ──────────
  get_prebuilt_trulens_recorder — TruLens recorder with Answer Relevance,
                                  Context Relevance, and Groundedness feedback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── LlamaIndex core (v0.10+) ────────────────────────────────────────────────
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes,
)
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever

# ── TruLens (v1.x — replaces the old trulens_eval package) ──────────────────
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as fOpenAI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SentenceWindowConfig:
    """Hyper-parameters for the Sentence-Window RAG pipeline."""

    embed_model: str = "local:BAAI/bge-small-en-v1.5"
    sentence_window_size: int = 3
    similarity_top_k: int = 6
    rerank_top_n: int = 2
    rerank_model: str = "BAAI/bge-reranker-base"
    save_dir: str = "sentence_index"


@dataclass
class AutoMergingConfig:
    """Hyper-parameters for the Auto-Merging RAG pipeline."""

    embed_model: str = "local:BAAI/bge-small-en-v1.5"
    chunk_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    similarity_top_k: int = 12
    rerank_top_n: int = 6
    rerank_model: str = "BAAI/bge-reranker-base"
    save_dir: str = "merging_index"


# ─────────────────────────────────────────────────────────────────────────────
# Index builders
# ─────────────────────────────────────────────────────────────────────────────


def build_sentence_window_index(
    documents: List[Document],
    llm,
    config: Optional[SentenceWindowConfig] = None,
) -> VectorStoreIndex:
    """
    Build (or load from disk) a VectorStoreIndex for Sentence-Window RAG.

    Each document sentence is stored as an individual node, but the node is
    augmented with a configurable window of surrounding sentences so the LLM
    has richer context during synthesis.

    Parameters
    ----------
    documents:
        List of LlamaIndex Document objects to index.
    llm:
        Instantiated LLM (e.g. OpenAI(model="gpt-4o-mini")).
    config:
        SentenceWindowConfig instance. Defaults to SentenceWindowConfig().

    Returns
    -------
    VectorStoreIndex
        The sentence-window vector index.
    """
    cfg = config or SentenceWindowConfig()
    save_path = Path(cfg.save_dir)

    # Apply global Settings (replaces the deprecated ServiceContext)
    Settings.llm = llm
    Settings.embed_model = cfg.embed_model
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=cfg.sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    if save_path.exists():
        logger.info("Loading existing sentence-window index from '%s'.", save_path)
        storage_context = StorageContext.from_defaults(persist_dir=str(save_path))
        sentence_index = load_index_from_storage(storage_context)
    else:
        logger.info("Building sentence-window index and persisting to '%s'.", save_path)
        sentence_index = VectorStoreIndex.from_documents(documents)
        sentence_index.storage_context.persist(persist_dir=str(save_path))

    return sentence_index


def build_automerging_index(
    documents: List[Document],
    llm,
    config: Optional[AutoMergingConfig] = None,
) -> VectorStoreIndex:
    """
    Build (or load from disk) a VectorStoreIndex for Auto-Merging RAG.

    Documents are chunked into a multi-level hierarchy (e.g. 2048 -> 512 -> 128
    tokens). Only leaf nodes are embedded; at query time, if the majority of a
    parent's children are retrieved the retriever automatically merges them into
    the larger parent chunk, reducing LLM calls while preserving context.

    Parameters
    ----------
    documents:
        List of LlamaIndex Document objects to index.
    llm:
        Instantiated LLM (e.g. OpenAI(model="gpt-4o-mini")).
    config:
        AutoMergingConfig instance. Defaults to AutoMergingConfig().

    Returns
    -------
    VectorStoreIndex
        The auto-merging vector index.
    """
    cfg = config or AutoMergingConfig()
    save_path = Path(cfg.save_dir)

    # Apply global Settings
    Settings.llm = llm
    Settings.embed_model = cfg.embed_model

    if save_path.exists():
        logger.info("Loading existing auto-merging index from '%s'.", save_path)
        storage_context = StorageContext.from_defaults(persist_dir=str(save_path))
        return load_index_from_storage(storage_context)

    logger.info("Building auto-merging index and persisting to '%s'.", save_path)

    # Build the hierarchical node tree
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=cfg.chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # All nodes (parents + leaves) live in the docstore so the retriever can
    # walk up the tree during merging; only leaf nodes are embedded.
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    automerging_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    automerging_index.storage_context.persist(persist_dir=str(save_path))

    return automerging_index


# ─────────────────────────────────────────────────────────────────────────────
# Query-engine factories
# ─────────────────────────────────────────────────────────────────────────────


def get_sentence_window_query_engine(
    sentence_index: VectorStoreIndex,
    config: Optional[SentenceWindowConfig] = None,
) -> RetrieverQueryEngine:
    """
    Wrap a sentence-window index with a post-processing pipeline.

    Post-processors applied in order:
      1. MetadataReplacementPostProcessor — swaps each retrieved sentence
         node's text with its stored surrounding window for richer LLM context.
      2. SentenceTransformerRerank — cross-encoder reranking to surface the
         most relevant chunks from the broader retrieval set.

    Parameters
    ----------
    sentence_index:
        A VectorStoreIndex created by build_sentence_window_index.
    config:
        SentenceWindowConfig instance. Defaults to SentenceWindowConfig().

    Returns
    -------
    RetrieverQueryEngine
    """
    cfg = config or SentenceWindowConfig()

    postprocessors = [
        MetadataReplacementPostProcessor(target_metadata_key="window"),
        SentenceTransformerRerank(top_n=cfg.rerank_top_n, model=cfg.rerank_model),
    ]

    return sentence_index.as_query_engine(
        similarity_top_k=cfg.similarity_top_k,
        node_postprocessors=postprocessors,
    )


def get_automerging_query_engine(
    automerging_index: VectorStoreIndex,
    config: Optional[AutoMergingConfig] = None,
) -> RetrieverQueryEngine:
    """
    Wrap an auto-merging index with retrieval and reranking.

    Retrieval pipeline:
      1. AutoMergingRetriever — retrieves leaf nodes and automatically merges
         them into parent chunks when a majority threshold is met.
      2. SentenceTransformerRerank — cross-encoder reranking over merged results.

    Parameters
    ----------
    automerging_index:
        A VectorStoreIndex created by build_automerging_index.
    config:
        AutoMergingConfig instance. Defaults to AutoMergingConfig().

    Returns
    -------
    RetrieverQueryEngine
    """
    cfg = config or AutoMergingConfig()

    base_retriever = automerging_index.as_retriever(
        similarity_top_k=cfg.similarity_top_k
    )
    retriever = AutoMergingRetriever(
        base_retriever,
        automerging_index.storage_context,
        verbose=True,
    )
    reranker = SentenceTransformerRerank(top_n=cfg.rerank_top_n, model=cfg.rerank_model)

    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[reranker],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TruLens evaluation
# ─────────────────────────────────────────────────────────────────────────────


def get_prebuilt_trulens_recorder(
    query_engine,
    app_name: str,
    app_version: str = "v1",
) -> TruLlama:
    """
    Create a TruLens TruLlama recorder with the full RAG-triad feedback suite.

    Feedback functions
    ------------------
    Answer Relevance
        Measures whether the final answer addresses the user query
        (input -> output).
    Context Relevance
        Measures how well each retrieved chunk relates to the query
        (input -> each source node).  Aggregated via np.mean.
    Groundedness
        Measures whether every statement in the answer is supported by the
        retrieved context (source nodes -> output).  Aggregated via np.mean.

    Parameters
    ----------
    query_engine:
        A LlamaIndex query engine to instrument.
    app_name:
        Human-readable name for this experiment in the TruLens dashboard.
    app_version:
        Version string for this experiment (default "v1").

    Returns
    -------
    TruLlama
        Configured recorder; use as a context manager around your query loop.

    Example
    -------
    >>> recorder = get_prebuilt_trulens_recorder(engine, "Sentence Window", "v1")
    >>> with recorder as recording:
    ...     for question in eval_questions:
    ...         engine.query(question)
    """
    provider = fOpenAI()
    context_selection = TruLlama.select_source_nodes().node.text

    # 1. Answer Relevance
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    # 2. Context Relevance
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons,
            name="Context Relevance",
        )
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    # 3. Groundedness
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
        )
        .on(context_selection)
        .on_output()
        .aggregate(np.mean)
    )

    return TruLlama(
        query_engine,
        app_name=app_name,
        app_version=app_version,
        feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
    )
