<div align="center">

# 🔍 Advanced RAG Application with LLMs

### Sentence-Window & Auto-Merging Retrieval · TruLens Evaluation · LlamaIndex v0.10+

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-v0.10%2B-FF6F61?style=for-the-badge)](https://docs.llamaindex.ai)
[![TruLens](https://img.shields.io/badge/TruLens-v1.x-7C3AED?style=for-the-badge)](https://www.trulens.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 📌 What This Project Does

This project implements and **benchmarks two advanced Retrieval-Augmented Generation (RAG) techniques** on top of a baseline RAG pipeline, using the **LlamaIndex** framework and **OpenAI LLMs**. All three pipelines are objectively evaluated using the **RAG Triad** (Answer Relevance, Context Relevance, Groundedness) via **TruLens**.

The core idea: standard RAG pipelines often retrieve too little context (narrow chunks) or too much noise (large chunks). This project explores two smarter retrieval strategies that adaptively surface the *right* amount of context.

---

## 🧠 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INPUT DOCUMENT(S)                    │
│                  Medical_Cost_Prediction.pdf             │
└─────────────────────────┬────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
   │  Baseline   │ │  Sentence   │ │    Auto-     │
   │    RAG      │ │   Window    │ │   Merging    │
   │  Pipeline   │ │  Retrieval  │ │  Retrieval   │
   └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
              ┌─────────────────────┐
              │   TruLens Eval      │
              │  (RAG Triad Score)  │
              │  · Answer Relevance │
              │  · Context Relevance│
              │  · Groundedness     │
              └─────────────────────┘
```

---

## 🔬 The Three Pipelines Explained

### 1️⃣ Baseline RAG Pipeline (`RAG_Pipeline.ipynb`)

The standard Ingestion → Retrieval → Synthesis pipeline:

| Stage | What Happens |
|-------|-------------|
| **Ingestion** | PDF is split into fixed-size chunks → embedded via `BAAI/bge-small-en-v1.5` → stored in a `VectorStoreIndex` |
| **Retrieval** | User query is embedded → Top-K most similar chunks are fetched |
| **Synthesis** | Retrieved chunks + query are passed to the LLM for a final answer |

> **Limitation:** Fixed-size chunks can cut sentences mid-thought, losing context at boundaries.

---

### 2️⃣ Sentence-Window Retrieval (`sentence_window_retrieval.ipynb`)

**Core idea:** Index at the *sentence* level (precision), but retrieve with *surrounding context* (recall).

```
Document → [S1] [S2] [S3] [S4] [S5] [S6] [S7]
                                             ↑ query matches S4
Retrieval window (size=3): [S3] [S4] [S5]  ← LLM sees this wider context
```

**How it works:**
1. `SentenceWindowNodeParser` splits each document into individual sentences
2. Each sentence node stores its surrounding ±N sentences in metadata as a `window`
3. At retrieval time, the `MetadataReplacementPostProcessor` replaces the narrow sentence with the full window
4. `SentenceTransformerRerank` (cross-encoder) re-scores the candidates for maximum precision

**Key parameter to tune:** `window_size` — more context vs. more noise trade-off.

---

### 3️⃣ Auto-Merging Retrieval (`automerging_retrieval.ipynb`)

**Core idea:** Build a *tree* of chunks. Retrieve small leaves (precision), but merge into parents when evidence converges (richer context, fewer LLM tokens).

```
Layer 1 (2048 tok): [════════════ Parent ════════════]
Layer 2  (512 tok): [══ Child-A ══] [══ Child-B ══] [══ Child-C ══]
Layer 3  (128 tok): [L1][L2][L3]   [L4][L5][L6]   [L7][L8][L9]
                     ↑ VectorStore only indexes leaf nodes (128-token)
```

**How it works:**
1. `HierarchicalNodeParser` creates a 3-level chunk tree (2048 → 512 → 128 tokens)
2. Only **leaf nodes** (128 tokens) are embedded in the `VectorStoreIndex`
3. `AutoMergingRetriever` checks: if enough sibling leaves are retrieved, it swaps them for their shared parent chunk
4. `SentenceTransformerRerank` re-scores the final merged result

**Key parameter to tune:** `chunk_sizes` — more layers = smaller leaves = lower LLM cost.

---

## 📊 Evaluation: The RAG Triad

All pipelines are evaluated using [TruLens](https://www.trulens.org/) with three feedback functions powered by an LLM-as-judge:

| Metric | Measures | Direction |
|--------|----------|-----------|
| **Answer Relevance** | Does the response actually answer the question? | input → output |
| **Context Relevance** | Are the retrieved chunks relevant to the question? | input → retrieved nodes |
| **Groundedness** | Is every claim in the answer supported by context? | retrieved nodes → output |

```python
# Run the TruLens dashboard to compare all experiments
tru.run_dashboard()  # Opens at http://localhost:8501
```

---

## 🗂️ Project Structure

```
RAG_Application_Using_LLM/
│
├── RAG_Pipeline.ipynb              # Baseline RAG — ingestion, retrieval, synthesis
├── sentence_window_retrieval.ipynb # Advanced: Sentence-Window technique + evaluation
├── automerging_retrieval.ipynb     # Advanced: Auto-Merging technique + evaluation
│
├── utils.py                        # Shared helper functions (indexes, query engines,
│                                   # TruLens recorder) — modern LlamaIndex v0.10+ API
│
├── Medical_Cost_Prediction.pdf     # Source document (used as the knowledge base)
├── eval_questions.txt              # 10 evaluation questions for TruLens benchmarking
└── default.sqlite                  # TruLens experiment results database
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- An OpenAI API key

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/RAG_Application_Using_LLM.git
cd RAG_Application_Using_LLM
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-huggingface \
            llama-index-postprocessor-flag-embedding-reranker \
            trulens trulens-apps-llamaindex trulens-providers-openai \
            sentence-transformers pypdf
```

### 4. Set your OpenAI API key
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

### 5. Run the notebooks
Open and run the notebooks in this order:
1. `RAG_Pipeline.ipynb` — establish the baseline
2. `sentence_window_retrieval.ipynb` — try Sentence-Window RAG
3. `automerging_retrieval.ipynb` — try Auto-Merging RAG

Then open the TruLens dashboard to compare results:
```python
from trulens.core import TruSession
session = TruSession()
session.run_dashboard()
```

---

## 🔧 Key API: `utils.py`

The `utils.py` module uses the **modern LlamaIndex v0.10+ `Settings` API** (the old `ServiceContext` is deprecated) and **TruLens v1.x** imports.

### Configuration via dataclasses

```python
from utils import SentenceWindowConfig, AutoMergingConfig

# Customize hyper-parameters with type-safe dataclasses
sw_config = SentenceWindowConfig(
    sentence_window_size=5,   # wider context window
    similarity_top_k=8,       # retrieve more candidates before reranking
    rerank_top_n=3,
)

am_config = AutoMergingConfig(
    chunk_sizes=[4096, 1024, 256],  # deeper tree = smaller leaves = lower cost
    similarity_top_k=15,
)
```

### Building indexes

```python
from llama_index.llms.openai import OpenAI
from utils import build_sentence_window_index, build_automerging_index

llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

sw_index = build_sentence_window_index(documents, llm, config=sw_config)
am_index = build_automerging_index(documents, llm, config=am_config)
```

### Getting query engines

```python
from utils import get_sentence_window_query_engine, get_automerging_query_engine

sw_engine = get_sentence_window_query_engine(sw_index, config=sw_config)
am_engine = get_automerging_query_engine(am_index, config=am_config)
```

### Running TruLens evaluation

```python
from trulens.core import TruSession
from utils import get_prebuilt_trulens_recorder

session = TruSession()
session.reset_database()

recorder = get_prebuilt_trulens_recorder(sw_engine, "Sentence Window", "v1")
with recorder as recording:
    for question in eval_questions:
        sw_engine.query(question)

session.run_dashboard()  # http://localhost:8501
```

---

## 💡 Techniques at a Glance

| Feature | Baseline RAG | Sentence-Window | Auto-Merging |
|---------|-------------|-----------------|--------------|
| **Chunk granularity** | Fixed-size | Per sentence | Hierarchical (multi-level) |
| **Context at retrieval** | Chunk only | Sentence + window | Leaf → auto-merges to parent |
| **Embedding level** | Chunk | Sentence | Leaf node |
| **Re-ranking** | ✗ | ✓ BGE cross-encoder | ✓ BGE cross-encoder |
| **Best for** | Quick prototyping | Dense, narrative text | Long docs, cost efficiency |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
