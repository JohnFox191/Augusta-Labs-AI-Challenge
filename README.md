# Augusta Labs AI Challenge

By: João Raposo

Hybrid RAG system for matching Portuguese companies with public incentives.

## Architecture

- **Database**: PostgreSQL + VectorChord (vchord) for high-performance vector similarity search
- **Embeddings**: Configurable via `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- **LLM**: Configurable via `CHAT_MODEL` (default: `gpt-4o-mini`)
- **API**: FastAPI with streaming SSE support
- **Vector Index**: VectorChord's for efficient ANN search
- 
## TODO | WHAT I WOULD ADD 
- Expand company data by performing scraping/querying data from aggregators (like Racius, eInforma) to extract how old the company is, aproximate revenue, location, etc. To better filter the elegibility criteria
- Do a better UI, ended up testing it with Swagger and Insomnia
- Setup a web search MCP so that the LLM could query the top-K companies' info to inform the matching
- Ended up discarding hard criteria due to the lack of tags and no time to set them up
- Add trigram index to the various fields of the incentives, to allow specific incentive search without using semantic search (adding the id fields into the embedding degraded performance)
- 
## Setup

### 1. Install Dependencies

Install conda and docker.

Setup the conda environment:
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate augustalabs-challenge
```

Setup Postgres and VectorChord by running:

```bash
# Setup vchord with appropriate password and parameters
docker run --name vchord-demo -e POSTGRES_PASSWORD=<password> -p 5432:5432 -d tensorchord/vchord-postgres:pg18-v0.5.3
```

### 2. Configure Environment

Create a `.env` file in the `AugustaLabsChallenge` directory:

```env
OPENAI_API_KEY=<api-key>
OPENAI_BASE_URL=https://api.openai.com/v1
DB_USER=<dbname>
DB_PASSWORD=<password>
DB_HOST=<url>
DB_PORT=<port>
DB_NAME=augusta_challenge_db_live
DATA_DIR=..
MATCHES_CSV_FILE=matches.csv
RUN_MATCHING_ON_START=false
DB_RESET_ON_START=true
USE_PRECOMPUTED_MATCHES=false
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=2000
MATCH_MULTIPLIER=5
SKIP_JSON_CHAT_REQUEST=true
```

### 3. Setup Database

Create tables and indexes:
```bash
python db_setup.py
```

This creates tables, indexes (VectorChord optimized HNSW for vectors, trigram for fuzzy text search), and loads CSVs.


**Fresh Start**: To completely reset the database:
```bash
DB_RESET_ON_START=true python db_setup.py
```

### 4. Generate Embeddings

```bash
python etl_and_embed.py
```

This batches embedding requests, and prepares embeddings for data to reduce API calls and cost. 

### 5. Run Matching (Optional)

```bash
RUN_MATCHING_ON_START=true python app.py
```

Or run matching separately:

```python
from app import run_company_matching
run_company_matching()
```

Outputs `matches.csv` with top 5 companies per incentive.

### 6. Start API Server

```bash
python app.py
```

Access interactive docs at `http://localhost:8000/docs`

## API Endpoints

### POST /chat

Non-streaming chat endpoint with performance metrics. Uses RAG pipeline: classify intent → retrieve context → synthesize answer.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Que incentivos existem para PMEs no setor de I&D?", "user_id": "optional-user-id"}'
```

**Response includes:**
- `answer`: Generated answer in Portuguese
- `intent`: Classified intent (`company_specific`, `company_general`, `incentive_specific`, `incentive_general`, `match_search`, `other`)
- `context_retrieved`: Raw context data used for answer generation
- `matching_mode`: "precomputed" or "runtime" (for match_search queries)
- `matching_time_ms`: Time taken for matching operation (if applicable)
- `match_direction`: "incentive_to_companies" (for match_search queries)
- `incentive_id`: Incentive ID when available (for CSV download)
- `total_time_ms`: Total request time

### POST /chat/stream

Streaming chat endpoint (faster first token via SSE). Same pipeline as `/chat` but streams response tokens.

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Quais as melhores empresas para o SIID?", "user_id": "optional-user-id"}'
```

**Response format:** Server-Sent Events (SSE) with `data:` prefixed JSON chunks containing `content` field.

### POST /matches

Get top 5 company matches for a specific incentive as JSON. Performs runtime matching using enhanced LLM-based filtering and preference sorting.

```bash
curl -X POST http://localhost:8000/matches \
  -H "Content-Type: application/json" \
  -d '{"incentive_id": 123}'
```

**Response includes:**
- `incentive_id`: The incentive ID requested
- `matches`: Array of top 5 company matches with:
  - `company_id`: Company database ID
  - `company_name`: Company name
  - `rank`: Ranking (1-5)
  - `eligibility_reason`: Brief explanation of eligibility
  - `has_website`: Boolean indicating if company has website
  - `semantic_score`: Vector similarity score
  - `preference_score`: Composite preference score (website + CAE + trade description)
- `matching_time_ms`: Total matching time in milliseconds

### GET /matches/csv

Download matches as CSV file. Can retrieve matches for a specific incentive or all matches.

```bash
# Get matches for specific incentive
curl -X GET "http://localhost:8000/matches/csv?incentive_id=123" -o matches_incentive_123.csv

# Get all matches
curl -X GET "http://localhost:8000/matches/csv" -o matches_all.csv
```

**CSV columns:**
- `incentive_id`: Incentive database ID
- `incentive_title`: Incentive title
- `company_id`: Company database ID
- `company_name`: Company name
- `rank`: Ranking (1-5)
- `preference_score`: Composite preference score
- `semantic_score`: Vector similarity score
- `eligibility_reason`: Brief explanation (runtime mode only)
- `has_website`: Boolean flag (runtime mode only)

## Configuration

All settings are configurable via environment variables:

**OpenAI Configuration:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_BASE_URL`: API endpoint (default: `https://api.openai.com/v1`) - change for Azure, local models, etc.
- `CHAT_MODEL`: LLM model for chat, criteria extraction, filtering, and sorting (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL`: Model for generating embeddings (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSIONS`: Vector dimensions - **must match your embedding model**:
  - `1536`: text-embedding-3-small, text-embedding-ada-002
  - `3072`: text-embedding-3-large
  - `768`: many open-source models (sentence-transformers, etc.)
  - Check your model's documentation for the correct value
- `EMBEDDING_BATCH_SIZE`: How many texts to embed per API call (default: `100`)
- `SKIP_JSON_CHAT_REQUEST`: Skip JSON format requests and use text format directly (default: `false`) - Set to `true` for local LLMs that don't support JSON mode

**Database Configuration:**
- `DB_USER`: PostgreSQL username (default: `postgres`)
- `DB_PASSWORD`: PostgreSQL password (default: `password`)
- `DB_HOST`: Database host (default: `localhost`)
- `DB_PORT`: Database port (default: `5432`)
- `DB_NAME`: Database name (default: `incentives_db`)

**Application Configuration:**
- `DATA_DIR`: Directory containing CSV files (default: parent directory)
- `MATCHES_CSV_FILE`: Output filename for matches CSV (default: `matches.csv`)
- `RUN_MATCHING_ON_START`: Auto-run matching when starting app (default: `false`)
- `USE_PRECOMPUTED_MATCHES`: Use cached matches from database (default: `true`) or runtime matching (`false`)
- `DROP_MATCHES_TABLE`: Drop and recreate matches table on startup (default: `false`)
- `DB_RESET_ON_START`: **WARNING**: Drop entire database and recreate from scratch (default: `false`)
- `MATCH_MULTIPLIER`: Multiplier for initial candidate pool size (default: `10`) - fetches `top_n * MATCH_MULTIPLIER` candidates before filtering
- `LOG_LEVEL`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

## Cost Optimization

- Embeddings are pre-computed and cached in PostgreSQL
- Batch embedding generation
- VectorChord's optimized HNSW indexes for fast vector search (no table scans)
- Cheap models: `gpt-4o-mini` and `text-embedding-3-small`

###  Data Assumptions

- Semantic search is primary matcher; hard filters are best-effort
- Trigram indexes help with fuzzy name/id search
