# Kapston RAG

An advanced Retrieval-Augmented Generation (RAG) system built over the
**Kapston Home Services** documentation.

The project was developed from the ground up to understand the internals
of RAG rather than relying on a high-level orchestration framework. It
evolved from basic PDF ingestion and dense vector search into a hybrid
retrieval pipeline combining **MiniLM embeddings, Qdrant, BM25,
Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and Gemini
generation**.

The system is available through both a **CLI** and a **Streamlit chat
interface**.

------------------------------------------------------------------------

## Description

Kapston RAG answers questions about Kapston Home Services using only the
supplied company documentation.

The project separates retrieval from generation:

-   **Retrieval** determines which pieces of documentation are relevant.
-   **Reranking** improves the ordering of retrieved candidates.
-   **Prompting** supplies only the selected evidence to the LLM.
-   **Gemini** is used solely for grounded answer generation.

The project also includes an evaluation set containing supported,
negative-supported, and out-of-scope questions. This was used to compare
chunking and retrieval strategies instead of optimizing the system
around individual example queries.

### Current retrieval pipeline

1.  Dense semantic retrieval using `all-MiniLM-L6-v2`
2.  BM25 lexical retrieval
3.  Reciprocal Rank Fusion
4.  Cross-Encoder reranking
5.  Top-5 context selection
6.  Grounded generation using Gemini

The selected production chunking strategy is:

-   **Chunk size:** 256 tokens
-   **Overlap:** 48 tokens

The production document index contains **32 chunks**.

------------------------------------------------------------------------

## Architecture

### Offline indexing

``` text
Kapston PDF
    │
    ▼
data_loader.py
    │
    ▼
chunker.py
256 tokens
48 overlap
    │
    ▼
embedder.py
all-MiniLM-L6-v2
    │
    ▼
indexer.py
    │
    ▼
Qdrant
kapston_documents_v2
```

### Online RAG pipeline

``` text
                         USER QUERY
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Dense Retrieval          BM25 Retrieval
          MiniLM + Qdrant             Top 20
              Top 20                    │
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                   Reciprocal Rank Fusion
                          RRF K = 60
                             │
                             ▼
                     Hybrid Top 20
                             │
                             ▼
                       Cross-Encoder
                         Reranking
                             │
                             ▼
                         Final Top 5
                             │
                             ▼
                         prompter.py
                             │
                             ▼
                         generator.py
                             │
                             ▼
                           Gemini
                             │
                             ▼
                           ANSWER
```

### Application interfaces

``` text
                    src/rag.py
                   KapstonRAG
                       ▲
              ┌────────┴────────┐
              │                 │
            Main.py            app.py
              │                 │
          Terminal          Streamlit
```

Both interfaces use the same RAG implementation.

------------------------------------------------------------------------

## How It Works

### 1. Document loading

`data_loader.py` reads the Kapston Home Services PDF using `pypdf` and
extracts text page by page.

Page metadata is preserved so retrieved evidence can later be traced
back to its source page.

### 2. Chunking

`chunker.py` tokenizes the extracted pages and splits them into chunks
of:

``` text
256 tokens
48-token overlap
```

Several strategies were evaluated before choosing this configuration,
including:

-   500-character chunks
-   128-token chunks
-   192-token chunks
-   256-token chunks
-   structure-aware chunks

The 256-token configuration produced the best overall retrieval results
on the project evaluation set.

### 3. Embedding

`embedder.py` uses:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

to create normalized dense embeddings.

The model produces **384-dimensional vectors**.

The same embedding module is shared by indexing and query retrieval so
document and query vectors always use the same model and normalization
behavior.

### 4. Indexing

`indexer.py` embeds the document chunks and stores them in the local
Qdrant collection:

``` text
kapston_documents_v2
```

Each point stores metadata such as:

``` text
source
page
chunk
page_chunk
text
```

The local Qdrant database is stored under:

``` text
storage/qdrant/
```

The storage directory is generated locally and is excluded from Git.

### 5. Dense retrieval

For each question, MiniLM creates a normalized query embedding.

Qdrant performs semantic vector search and retrieves up to:

``` text
DENSE_TOP_K = 20
```

A provisional dense similarity threshold of:

``` text
MIN_RELEVANCE_SCORE = 0.20
```

is applied before fusion.

This threshold is a retrieval filter, not a calibrated answerability
probability.

### 6. BM25 retrieval

A lightweight BM25 implementation provides lexical retrieval over the
same document chunks.

Configuration:

``` text
BM25_TOP_K = 20
BM25_K1 = 1.5
BM25_B = 0.75
```

BM25 complements dense retrieval by rewarding exact lexical matches that
semantic embeddings may rank poorly.

### 7. Reciprocal Rank Fusion

Dense and BM25 scores are not directly comparable because they use
different scoring systems.

Instead, the project combines their rankings using Reciprocal Rank
Fusion:

``` text
RRF(document) =
    1 / (K + dense_rank)
  + 1 / (K + bm25_rank)
```

with:

``` text
RRF_K = 60
```

The fused ranking produces up to **20 hybrid candidates**.

### 8. Cross-Encoder reranking

The hybrid candidates are reranked using:

``` text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike the bi-encoder used for dense retrieval, the Cross-Encoder
jointly evaluates the query and candidate text.

Only the final:

``` text
FINAL_TOP_K = 5
```

chunks are sent to the generation stage.

### 9. Prompt construction

`prompter.py` converts the question and retrieved chunks into a grounded
prompt.

The generator is instructed to:

-   answer using only the supplied context,
-   avoid inventing information,
-   avoid relying on outside knowledge,
-   explain explicitly documented unavailable information when
    appropriate,
-   refuse when the retrieved documentation does not support an answer,
-   reference relevant page numbers.

### 10. Generation

`generator.py` sends the grounded prompt to Gemini.

Gemini is used **only for generation**.

Retrieval, lexical search, fusion, and reranking are handled
independently of the LLM.

------------------------------------------------------------------------

## Design Choices

### Why build the retrieval pipeline manually?

The project intentionally avoided starting with LangChain or another
high-level RAG framework.

The goal was to understand the underlying components directly:

``` text
Chunking
   ↓
Embeddings
   ↓
Vector Search
   ↓
Lexical Search
   ↓
Rank Fusion
   ↓
Reranking
   ↓
Prompting
   ↓
Generation
```

This makes retrieval behavior easier to inspect, evaluate, and debug.

### Why 256-token chunks?

Chunking strategies were evaluated against the project question set.

  Strategy            Chunks   Evidence Recall@5
  ----------------- -------- -------------------
  char_500                83              86.11%
  token_128               62              88.89%
  token_192               40              88.89%
  **token_256**       **32**          **91.67%**
  structure-aware         51              83.33%

The 256-token strategy provided the strongest overall result for this
document, embedding model, and evaluation set.

It should not be interpreted as a universally optimal chunk size.

### Why hybrid retrieval?

Dense retrieval is good at semantic similarity but can miss exact
terminology.

BM25 is strong at lexical matching but weaker when the query and
document use different wording.

Combining them improves candidate coverage:

``` text
Dense → semantic relevance
BM25  → lexical relevance
```

### Why RRF?

Cosine similarity and BM25 scores have different numerical scales.

RRF combines **rank positions** rather than raw scores, avoiding manual
score normalization.

### Why a Cross-Encoder?

Dense retrieval uses a bi-encoder because it is efficient: document
embeddings can be precomputed.

A Cross-Encoder is more expensive because it jointly evaluates each
query-document pair, but it can perform more detailed relevance ranking.

Therefore it is applied only after candidate generation.

### Why Top-20 candidates but only Top-5 context?

A wider candidate pool improves the chance that useful evidence is
discovered.

The Cross-Encoder then reduces this pool to a smaller set of
higher-ranked chunks before generation.

This balances:

-   candidate recall,
-   context quality,
-   inference cost,
-   prompt size,
-   noise.

### Why keep Gemini generation-only?

The project maintains a clear separation between retrieval and
generation.

``` text
Retrieval system → decides evidence
Gemini           → writes the answer
```

No LLM-based query expansion, LLM reranking, LLM-as-judge, or separate
LLM answerability classifier is used in the final architecture.

### Why version the Qdrant collection?

Changing chunking strategies changes the indexed document
representation.

Instead of destructively modifying the original experimental collection,
the production index uses:

``` text
kapston_documents_v2
```

This makes index changes easier to reason about and verify.

### Evaluation-driven decisions

The evaluation set contains **49 questions**:

``` text
30 supported
 6 negative-supported
13 out-of-scope
```

A `negative_supported` question is one where the documentation
explicitly states that the requested detail is unavailable or
unspecified.

An `out_of_scope` question is one where the documentation contains
neither the answer nor explicit evidence of its absence.

One important result from the selected hybrid pipeline was:

``` text
Hybrid Evidence Candidate Recall@20: 36/36 (100%)
Final Cross-Encoder Evidence Recall@5: 33/36 (91.67%)
```

This distinction is important: candidate generation can successfully
find evidence even when final ranking does not place it in the Top 5.

------------------------------------------------------------------------

## Setup

### Prerequisites

-   Python
-   Git
-   A Gemini API key

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd Kapston-RAG
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file in the project root.

You can copy `.env.example`:

``` bash
copy .env.example .env
```

Then set:

``` env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit `.env`.

### 5. Add the source document

Place the Kapston Home Services documentation PDF in:

``` text
data/
```

The current loader expects the configured Kapston documentation filename
used by the project.

### 6. Build the vector index

For a fresh clone, run:

``` bash
python src/indexer.py
```

This creates the local Qdrant database under:

``` text
storage/qdrant/
```

The current indexer protects the existing production collection from
accidental overwrite. If the collection already exists, do not rebuild
it unless you intentionally change the indexing configuration.

### 7. Run CLI mode

``` bash
python Main.py
```

### 8. Run the Streamlit UI

``` bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal and
normally open it in your browser.

> **Local Qdrant note:** the project currently uses embedded/local
> Qdrant storage. Do not run the CLI and Streamlit application
> simultaneously because separate processes cannot concurrently open the
> same local Qdrant storage directory. Stop one process before starting
> the other.

------------------------------------------------------------------------

## Project Structure

``` text
Kapston-RAG/
│
├── app.py
│   └── Streamlit chat interface
│
├── Main.py
│   └── Command-line interface
│
├── requirements.txt
├── .gitignore
├── .env.example
├── .env                     # ignored
│
├── data/
│   └── Kapston documentation PDF
│
├── evaluation/
│   ├── questions.json
│   └── experimental/evaluation scripts
│
├── src/
│   ├── data_loader.py
│   │   └── PDF → pages
│   │
│   ├── chunker.py
│   │   └── pages → 256-token chunks
│   │
│   ├── embedder.py
│   │   └── text/query → MiniLM embeddings
│   │
│   ├── indexer.py
│   │   └── chunks + vectors → Qdrant
│   │
│   ├── retriever.py
│   │   └── Dense + BM25 + RRF + Cross-Encoder
│   │
│   ├── prompter.py
│   │   └── retrieved evidence → grounded prompt
│   │
│   ├── generator.py
│   │   └── prompt → Gemini response
│   │
│   └── rag.py
│       └── reusable KapstonRAG service
│
└── storage/
    └── qdrant/               # generated locally / ignored
```

------------------------------------------------------------------------

## Example Questions

### Company information

``` text
What is the full name of Kapston Home Services?
```

``` text
What is Kapston's brand statement?
```

``` text
How can customers book Kapston services?
```

### Services

``` text
What home cleaning services does Kapston provide?
```

``` text
What plumbing services are available?
```

``` text
What AC services does Kapston provide?
```

The AC-services question is also a useful known retrieval/ranking test
case from development.

### Refunds and policies

``` text
What is Kapston's refund policy?
```

``` text
How does cancellation work?
```

``` text
Are Kapston professionals independent contractors?
```

### Explicitly unavailable information

``` text
What training curriculum does Kapston use for its professionals?
```

For questions like this, the documentation may explicitly state that the
requested information is not publicly provided. The system can therefore
return a grounded negative answer rather than treating the question as
completely unrelated.

### Out-of-scope questions

``` text
How many employees does Kapston currently have?
```

``` text
What is Kapston's GST number?
```

``` text
What percentage commission does Kapston charge professionals?
```

If the retrieved documentation does not support the requested
information, the generator is instructed not to invent an answer.

------------------------------------------------------------------------

## Configuration

The primary retrieval configuration is defined in the project source
modules.

### Chunking

``` text
CHUNK_SIZE = 256
OVERLAP = 48
```

### Embeddings

``` text
Model:
all-MiniLM-L6-v2

Vector size:
384

Normalized embeddings:
True
```

### Qdrant

``` text
Collection:
kapston_documents_v2

Local storage:
storage/qdrant

Distance:
Cosine
```

### Dense retrieval

``` text
DENSE_TOP_K = 20
MIN_RELEVANCE_SCORE = 0.20
```

The relevance threshold is provisional and should not be interpreted as
a calibrated confidence or answerability score.

### BM25

``` text
BM25_TOP_K = 20
BM25_K1 = 1.5
BM25_B = 0.75
```

### Reciprocal Rank Fusion

``` text
RRF_K = 60
HYBRID_TOP_K = 20
```

### Cross-Encoder

``` text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Final context

``` text
FINAL_TOP_K = 5
```

### Generation

Gemini is configured in `generator.py`.

The API key is read from:

``` env
GEMINI_API_KEY
```

through the project's `.env` file.

------------------------------------------------------------------------

## Known Limitations

-   Final retrieval Recall@5 is not perfect even though hybrid candidate
    Recall@20 reached 100% on the answerable evaluation questions.
-   AC services remains a known final-ranking failure in the evaluated
    configuration.
-   Negative-supported evidence can sometimes be ranked poorly by the
    Cross-Encoder.
-   The current BM25 tokenizer is intentionally simple.
-   There is no dedicated answerability classifier.
-   The current Qdrant setup uses local embedded storage and therefore
    should not be opened concurrently by separate CLI and Streamlit
    processes.
-   Evaluation results are specific to this document, model
    configuration, and 49-question evaluation set.

------------------------------------------------------------------------

## Key Takeaway

The project treats RAG as a pipeline rather than a single model:

``` text
Source Document
      ↓
Chunking
      ↓
Embeddings
      ↓
Candidate Retrieval
      ↓
Rank Fusion
      ↓
Reranking
      ↓
Grounded Context
      ↓
Generation
      ↓
Answer
```
