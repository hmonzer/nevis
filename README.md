# Nevis API

A document management and semantic search API built with FastAPI, designed for wealth management use cases. Nevis provides hybrid search capabilities combining vector similarity and keyword matching across clients and documents, with optional LLM-powered document summarization.

## Out Of Scope
Some features that would be necessary in a production environment are not included in this project:
- Authentication & Authorization:
  - Scoping requests to specific wealth managers. Entities and APIs are not linked to wealth managers.
- More Robust DB migrations:
  - Currently, the database schema is created on startup. No migration scripts are generated/applied.
- _[Will add more points once I think of them]_

## Tradeoffs

**Local Embedding Models**: This service uses small, locally-run models from SentenceTransformers (`all-MiniLM-L6-v2` for embeddings, `ms-marco-MiniLM-L-6-v2` for reranking). This design choice prioritizes:

- **Simplicity**: No external API dependencies for core search functionality
- **Cost**: No per-request charges for embedding generation
- **Latency**: No network round-trips to external services

However, this comes with a **tradeoff on semantic accuracy**. Larger models (e.g., OpenAI's `text-embedding-3-large`, Cohere's `embed-v3`, or larger open-source models like `bge-large`) typically provide better semantic understanding, especially for domain-specific content.

**Extending to External Models**: The application is designed with this extensibility in mind. To use larger externally-hosted models:
1. Implement the `EmbeddingService` interface in `src/app/core/services/embedding.py`
2. Create a new provider (e.g., `OpenAIEmbedding`, `CohereEmbedding`)
3. Register it in the dependency injection container (`src/app/containers.py`)

Similarly, the `RerankerService` interface can be extended for more powerful reranking models.
> In case the model generates different embedding dimensions and contet window, some additional changes are required such as modifying the chunking strategy and vector column in the DB


- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker Compose](#docker-compose)
- [API Endpoints](#api-endpoints)
- [Document Upload & Processing Flow](#document-upload--processing-flow)
- [Search Architecture](#search-architecture)
  - [Hybrid Search Pipeline](#hybrid-search-pipeline)
  - [Client Search](#client-search)
  - [Document Search](#document-search)
  - [Reranking](#reranking)
- [Configuration](#configuration)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Evaluation Tests](#evaluation-tests)
  - [Evaluation Metrics](#evaluation-metrics)

## Architecture Overview

Nevis follows a **Clean Architecture** pattern with clear separation of concerns:

```
+-------------------------------------------------------------+
| API Layer (FastAPI)                                         |
| - Routes (src/app/api/v1/*.py)                              |
| - Request/Response Schemas (src/client/schemas.py)          |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
| Domain Layer                                                |
| - Domain Models (src/app/core/domain/models.py)             |
| - Services (src/app/core/services/*.py)                     |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
| Infrastructure Layer                                        |
| - Database Entities (src/app/infrastructure/entities/)      |
| - Repositories (src/app/infrastructure/*_repository.py)     |
| - Mappers (src/app/infrastructure/mappers/)                 |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
| Shared Layer                                                |
| - Database, UnitOfWork, Base Classes (src/shared/)          |
| - S3 Blob Storage (src/shared/blob_storage/)                |
+-------------------------------------------------------------+
```

## Project Structure

```
nevis/
├── src/
│   ├── app/
│   │   ├── api/v1/           # API endpoints (clients, documents, search)
│   │   ├── core/
│   │   │   ├── domain/       # Domain models (Client, Document, DocumentChunk)
│   │   │   └── services/     # Business logic services
│   │   ├── infrastructure/   # Database entities, repositories, mappers
│   │   ├── config.py         # Application settings
│   │   ├── containers.py     # Dependency injection container
│   │   └── main.py           # FastAPI application entry point
│   ├── client/               # API client and schemas
│   └── shared/               # Database, UnitOfWork, S3 storage
├── tests/
│   ├── app/                  # Unit and integration tests
│   ├── e2e_eval/             # End-to-end evaluation tests
│   └── integration/          # Integration tests
├── docker-compose.yml        # Docker services configuration
├── Dockerfile                # Application container
└── pyproject.toml            # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker and Docker Compose (for containerized setup)
- PostgreSQL with pgvector extension (or use Docker)


### Local Development

1. **Clone the repository and install dependencies:**

```bash
git clone <repository-url>
cd nevis
uv sync
```

2. **Set up environment variables:**

Create a `.env` file in the project root:

```env
# More ENV variables can be found in src/app/config.py

# Optional: Enable LLM summarization
SUMMARIZATION_ENABLED=true
SUMMARIZATION_PROVIDER=claude  # or "gemini"
ANTHROPIC_API_KEY=your-api-key  # for Claude
# GOOGLE_API_KEY=your-api-key   # for Gemini
```
3. **Docker Compose**

Run the complete stack with Docker Compose:

```bash
# Build and start all services
docker compose up --build
# Stop all services
docker compose down
```

> [!NOTE]
> **Startup Delay**: The application loads ML models (embedding and reranker) during startup to ensure fast API responses. On first run, models are downloaded from HuggingFace (~400MB total), which may take a few seconds/minutes depending on your connection.

**Services included:**
- `app` - Nevis API (port 8000)
- `db` - PostgreSQL with pgvector (port 5432)
- `localstack` - S3-compatible storage (port 4566)

**Development with live reload:**

Docker Compose is configured with `watch` for automatic syncing:
- Code changes in `./src` are synced to the container
- Changes to `pyproject.toml` trigger a rebuild


The API will be available at `http://localhost:8000`.**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/clients/` | POST | Create a new client |
| `/api/v1/clients/{client_id}` | GET | Get client by ID |
| `/api/v1/clients/{client_id}/documents/` | POST | Upload a document for a client |
| `/api/v1/clients/{client_id}/documents/` | GET | List all documents for a client |
| `/api/v1/clients/{client_id}/documents/{document_id}` | GET | Get a specific document |
| `/api/v1/clients/{client_id}/documents/{document_id}/download` | GET | Get pre-signed S3 URL for document download |
| `/api/v1/search/` | GET | Unified search across clients and documents |
| `/health` | GET | Health check endpoint |

**Example: Create a client**
```bash
curl -X POST http://localhost:8000/api/v1/clients/ \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "description": "High net worth investor interested in tech stocks"
  }'
```

**Example: Search**
```bash
curl "http://localhost:8000/api/v1/search/?q=tech%20investor&top_k=5"
```

## Document Upload & Processing Flow

When a document is uploaded, it goes through the following pipeline:

```
                         DOCUMENT UPLOAD FLOW
==============================================================================

  1. API Request            2. S3 Upload            3. Create Record
  +-------------+          +-------------+          +-------------+
  | POST        |          | Upload to   |          | Document    |
  | /documents/ | -------> | S3 Bucket   | -------> | Status:     |
  | {title,     |          | (LocalStack |          | PENDING     |
  |  content}   |          |  or AWS)    |          |             |
  +-------------+          +-------------+          +-------------+
                                                          |
                                                          v
  +-----------------------------------------------------------------------+
  |                    BACKGROUND PROCESSING                               |
  +-----------------------------------------------------------------------+
                                                          |
  4. Chunking               5. Embedding            6. Summarization
  +-------------+          +-------------+          +-------------+
  | Recursive   |          | Generate    |          | LLM Summary |
  | Text        | -------> | Embeddings  | -------> | (Optional)  |
  | Splitter    |          | (MiniLM)    |          | Claude/     |
  | 256 tokens  |          | 384 dims    |          | Gemini      |
  +-------------+          +-------------+          +-------------+
                                                          |
                                                          v
                                                   +-------------+
                                                   | Document    |
                                                   | Status:     |
                                                   | PROCESSED   |
                                                   +-------------+
```

**Processing Steps:**

1. **Chunking**: Documents are split using `RecursiveChunkingStrategy` with token-based splitting (default: 256 tokens, 25 token overlap). The chunking uses the embedding model's tokenizer for accurate token counting and splits on sentence boundaries (`\n`, `. `) to preserve semantic coherence within chunks.
2. **Embedding**: Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
3. **Summarization** (optional): If enabled, an LLM generates a document summary
4. **Persistence**: Chunks with embeddings are stored in PostgreSQL with pgvector

> **Enabling Summarization**: To enable LLM-powered document summaries, add the appropriate API key to your `.env` file:
> - For Claude: `ANTHROPIC_API_KEY=your-key`
> - For Gemini: `GOOGLE_API_KEY=your-key`
>
> Also set `SUMMARIZATION_PROVIDER` to either `claude` or `gemini`. Without a valid API key, summarization is silently skipped.

## Search Architecture

Nevis implements a sophisticated hybrid search pipeline that combines multiple search strategies:

### Hybrid Search Pipeline

```
                              SEARCH PIPELINE
==============================================================================

                              +-------------+
                              |   Query     |
                              | "tax plan"  |
                              +------+------+
                                     |
                    +----------------+----------------+
                    v                                 v
           +---------------+                 +---------------+
           | CLIENT SEARCH |                 | DOCUMENT SEARCH|
           | (Fuzzy Match) |                 | (Hybrid)      |
           +-------+-------+                 +-------+-------+
                   |                                 |
                   |                    +------------+------------+
                   |                    v                         v
                   |           +---------------+         +---------------+
                   |           | VECTOR SEARCH |         |KEYWORD SEARCH |
                   |           | (Semantic)    |         | (Lexical)     |
                   |           |               |         |               |
                   |           | pgvector      |         | pg_trgm +     |
                   |           | cosine sim    |         | ts_rank       |
                   |           +-------+-------+         +-------+-------+
                   |                   |                         |
                   |                   +-----------+-------------+
                   |                               v
                   |                       +---------------+
                   |                       |     RRF       |
                   |                       | (Rank Fusion) |
                   |                       |   k = 60      |
                   |                       +-------+-------+
                   |                               |
                   |                               v
                   |                       +---------------+
                   |                       |   RERANKER    |
                   |                       | CrossEncoder  |
                   |                       | (ms-marco)    |
                   |                       +-------+-------+
                   |                               |
                   +----------------+--------------+
                                    v
                           +---------------+
                           | UNIFIED       |
                           | RESULTS       |
                           | (Sorted by    |
                           |  score)       |
                           +---------------+
```

### Client Search

Clients are searched using **fuzzy text matching** powered by PostgreSQL's `pg_trgm` extension:

- Searches across: email, first name, last name, description
- Returns trigram similarity scores
- Fast and effective for finding clients by partial matches

### Document Search

Documents use a **hybrid search** approach combining:

1. **Vector Search (Semantic)**
   - Query is embedded using the same model as documents
   - Cosine similarity search via pgvector
   - Captures semantic meaning ("retirement planning" matches "401k strategies")

2. **Keyword Search (Lexical)**
   - PostgreSQL full-text search with `ts_rank`
   - Trigram similarity for fuzzy matching
   - Captures exact terms and typo tolerance

3. **Reciprocal Rank Fusion (RRF)**
   - Combines results from both search methods
   - Formula: `RRF_score(d) = sum of 1/(k + rank_i(d))` where k=60
   - No score normalization needed
   - Documents appearing in both lists get boosted

### Reranking

After initial retrieval, results are **reranked** using a CrossEncoder model:

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores query-document pairs directly (more accurate than bi-encoder similarity)
- Applied to top candidates from RRF fusion
- Produces final relevance ordering

## Configuration

Key settings in `src/app/config.py` (configurable via environment variables using `__` as nested delimiter):

| Setting                     | Default | Description |
|-----------------------------|---------|-------------|
| `DATABASE_URL`              | `postgresql+asyncpg://localhost/nevis` | Database connection string |
| `S3__BUCKET_NAME`           | `nevis-documents` | S3 bucket for document storage |
| `S3__ENDPOINT_URL`          | `None` | S3 endpoint (set for LocalStack) |
| `EMBEDDING__MODEL_NAME`     | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RERANKER__MODEL_NAME`      | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `CHUNKING__SIZE`            | `256` | Document chunk size in tokens |
| `CHUNKING__OVERLAP`         | `25` | Overlap between chunks in tokens |
| `CHUNK_SEARCH__RERANKER_SCORE_THRESHOLD` | `2.0` | Min cross-encoder score for document chunks |
| `CLIENT_SEARCH__RERANKER_SCORE_THRESHOLD` | `1.5` | Min cross-encoder score for clients |
| `SUMMARIZATION__ENABLED`    | `true` | Enable LLM summarization |
| `SUMMARIZATION__PROVIDER`   | `claude` | LLM provider (`claude` or `gemini`) |

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/app/api/test_search_api.py

# Run with coverage
uv run pytest --cov=src
```

### Evaluation Suite

Nevis includes a standalone evaluation module (`src/eval`) that measures search quality using standard IR metrics. The evaluation can run against any Nevis API instance - local development, Docker, or remote deployments.

**Running the evaluation against Docker Compose:**

```bash
# 1. Start the application with Docker Compose
docker compose up --build -d

# 2. Wait for the app to be ready (check health endpoint)
curl http://localhost:8000/health

# 3. Run the evaluation suite
uv run python -m src.eval --url http://localhost:8000
```

**CLI Options:**

```bash
# View all available options
uv run python -m src.eval --help

# Run with custom evaluation data file
uv run python -m src.eval --url http://localhost:8000 --data path/to/eval_data.json

# Run with higher top-k (retrieve more results per query)
uv run python -m src.eval --url http://localhost:8000 --top-k 10

# Enable debug mode for full stack traces
uv run python -m src.eval --url http://localhost:8000 --debug
```

**Running via pytest (for CI/local testing):**

```bash
uv run pytest tests/e2e_eval/test_wealth_manager_eval.py -v -s
```

**Evaluation process:**
1. Loads synthetic corpus (clients + documents) from the data file
2. Ingests all data into the target Nevis API instance
3. Runs predefined queries with expected results
4. Compares retrieved results against ground truth
5. Calculates IR metrics using the `ranx` library
6. Reports per-use-case and aggregate metrics

### Evaluation Metrics

The evaluation suite measures three key Information Retrieval metrics:

| Metric | Description | Score Range |
|--------|-------------|-------------|
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank of first relevant result | 0.0 - 1.0 |
| **Recall@5** | Fraction of relevant items found in top 5 | 0.0 - 1.0 |
| **NDCG@5** | Normalized Discounted Cumulative Gain at 5 | 0.0 - 1.0 |

**Current Performance Benchmarks:**

```
Basic Eval Set
📊 METRICS BY USE CASE
================================================================================
Use Case                          |      MRR |   Recall@5 |   NDCG@5 | Precision
--------------------------------------------------------------------------------
Onboarding & KYC Retrieval        |   1.0000 |     1.0000 |   1.0000 |   1.0000
Investment Policy & Compliance    |   1.0000 |     1.0000 |   0.9532 |   1.0000
Client Interactions & Life Events |   1.0000 |     1.0000 |   0.9532 |   1.0000
Estate Planning & Trusts          |   1.0000 |     1.0000 |   1.0000 |   1.0000
Tax & Reporting                   |   1.0000 |     1.0000 |   1.0000 |   1.0000
--------------------------------------------------------------------------------
AVERAGE                           |   1.0000 |     1.0000 |   0.9901 |   1.0000
================================================================================

 Large Data set
📊 METRICS BY USE CASE
================================================================================
Use Case                     |      MRR |   Recall@5 |   NDCG@5 | Precision
---------------------------------------------------------------------------
Sterling Tech & Assets       |   0.6667 |     0.5000 |   0.5867 |   0.6667
Hawthorne Trust Governance   |   1.0000 |     1.0000 |   0.9532 |   1.0000
Rodriguez Executive Planning |   1.0000 |     0.8333 |   0.9201 |   1.0000
Cross-Client Tax & Legal     |   0.6667 |     0.6667 |   0.6667 |   0.6667
---------------------------------------------------------------------------
AVERAGE                      |   0.8333 |     0.7500 |   0.7817 |   0.8333
===========================================================================
```

### Understanding Metric Variations

The metrics above show perfect scores on the **basic evaluation set** but lower scores on the **large dataset**. This is expected behavior and reflects the limitations of the small, locally-run models used in this implementation.

**Why metrics are lower on larger/complex datasets:**

1. **Small Embedding Model**: The `all-MiniLM-L6-v2` model (22M parameters, 384 dimensions) provides good general-purpose embeddings but has limited semantic understanding compared to larger models. It may not capture subtle domain-specific relationships (e.g., "crypto allocation" vs "crypto assets").

2. **Small Cross-Encoder**: The `ms-marco-MiniLM-L-6-v2` reranker (22M parameters) is trained on web search data and may not recognize domain-specific relevance signals in wealth management terminology.

3. **Chunking Boundaries**: With a 256-token chunk size, relevant information can be split across chunks. This affects retrieval when query terms span multiple sections of a document.

**Improving Search Quality:**

To achieve higher metrics on complex datasets, consider:

| Approach | Model Examples | Expected Improvement |
|----------|----------------|---------------------|
| Larger embedding models | `bge-large-en-v1.5`, `text-embedding-3-large` | Better semantic understanding |
| Domain-tuned models | Fine-tuned on financial/legal text | Improved domain relevance |
| Larger cross-encoders | `ms-marco-MiniLM-L-12-v2`, `bge-reranker-large` | More accurate reranking |
| Hybrid approaches | Combine with BM25, query expansion | Better recall |

**For this implementation scope**, the current setup with small local models is sufficient and maintains simplicity. The architecture is designed to be extensible - swapping to larger models requires only implementing the `EmbeddingService` or `RerankerService` interfaces and updating the dependency injection container.

**Test Data:**

The evaluation uses `tests/e2e_eval/data/synthetic_wealth_data.json` containing:
- 6 wealth management clients with realistic profiles
- Multiple document types per client (passports, risk questionnaires, meeting notes, etc.)
- Query scenarios testing different search use cases
