# Semantic Anonymizer

Adversarial, LLM-powered anonymizer: a **Defender** rewrites text to hide sensitive attributes, an
**Attacker** tries to infer them back, and a **Judge** rules on **both** privacy (did anything still leak?)
and utility (is the rewrite still useful?). The system loops until the Judge passes the text as *private and useful*.

Built with **LangGraph** (workflow) + **LangChain** (LLM-agnostic). Default model: **`google_vertexai:gemini-2.5-flash`**.

---

## Build & run (Docker — recommended)

`docker compose up --build` builds and starts **two containers**: an **nginx** frontend (serves the UI and
proxies `/api` to the backend) and a **FastAPI + LangGraph** backend that calls the LLM.

```mermaid
flowchart LR
    U["Browser<br/>localhost:8080"]
    N["nginx<br/>(frontend container)"]
    B["FastAPI + LangGraph<br/>(backend container)"]
    V["Google Vertex AI<br/>gemini-2.5-flash"]
    C["config.yaml + .env<br/>(credentials)"]

    U <-->|"HTTP — UI + result"| N
    N -->|"/api proxy — streaming NDJSON"| B
    B -->|"LangChain init_chat_model"| V
    C -.->|"config"| B
```

**Prereqs:** Docker + Docker Compose.

```bash
cp .env.example .env          # then edit .env and set your Google Cloud project ID
docker compose up --build     # builds + starts both containers
```

> **Note:** Docker deployment with Vertex AI may require additional credential mounting (e.g., mounting the ADC JSON file or using a service account key). This is not yet fully configured in the Docker setup.

Open **http://localhost:8080**, paste a sentence, choose attributes to hide, hit **Anonymize** — the
Defender → Attacker → Judge flow streams round by round, then the final anonymized text appears.

Stop with `Ctrl+C`; fully remove with `docker compose down`.

---

## Build & run (local, without Docker)

**Prereqs:** [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv
manages the virtualenv and the Python version (3.10+) for you.

### Authentication (Google Vertex AI)

The default provider is Google Vertex AI. Authenticate locally using Application Default Credentials (ADC):

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-google-cloud-project-id>
```

Set the required environment variables in `.env` (copy from `.env.example`):

```bash
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

> **Security warning:** Do not commit `.env` or credential files. The `.env` file is git-ignored. The `.env.example` file is safe because it only contains placeholders.

### Alternative: OpenAI

To use OpenAI instead of Vertex AI:

1. Uncomment and set `OPENAI_API_KEY` in `.env`
2. Change the model strings in `backend/config.yaml` from `google_vertexai:gemini-2.5-flash` to `openai:gpt-4o` (or another OpenAI model)

**Backend**
```bash
cd backend
uv sync                                                # creates .venv + installs deps from uv.lock
uv run uvicorn app.main:app --reload --port 8000
```

**Frontend** — serve the static files with any web server:
```bash
cd frontend && python -m http.server 8080
```
Since the page is now on a different origin than the API, point it at the backend by adding this line
just before `<script src="app.js">` in `index.html`:
```html
<script>window.API_BASE = "http://localhost:8000";</script>
```
Then open http://localhost:8080. (CORS is enabled on the backend for dev. Under Docker this step isn't
needed — nginx proxies `/api` on the same origin.)

---

## Running tests

The project includes a pytest unit test suite (65 tests) that does not require LLM/API calls.

**Install dev dependencies and run tests:**

```bash
cd backend
uv sync --extra dev                         # installs pytest
uv run python -m pytest tests -v            # runs all tests
```

Or with pip:

```bash
pip install -e "backend[dev]"
python -m pytest backend/tests -v
```

The tests cover:
- `matcher.py` — ground_truth validation functions
- `scoring.py` — candidate scoring and feedback builders
- `ner.py` — NER/regex detection (regex-only, no spaCy model required)
- `llm.py` — `safe_structured_invoke()` with mocked chains

---

## Configuration

All settings live in **one file**: [backend/config.yaml](backend/config.yaml). Credentials are set via `.env`.

- **Swap the LLM / provider** — change the `"provider:model"` strings in `config.yaml`, e.g.
  `google_vertexai:gemini-2.5-flash` → `openai:gpt-4o` or `ollama:llama3.1` (LangChain handles the rest; install
  the matching `langchain-*` package and set its credentials). Or override all roles at once with the `MODEL` env var.
- **Tune the loop** — `max_iters` and the utility PASS thresholds (`min_task_utility`, `min_factual`,
  `min_format`). The leak verdict itself is made by the Judge LLM, so there is no confidence threshold to tune.

---

## How it works

The whole run is one LangGraph: the **Defender** rewrites, the **Attacker** attacks, then the **Judge**
checks **privacy first** — and only if nothing leaked does it score utility. It loops until **PASS**
(private *and* useful) or gives up after `max_iters` and returns the best attempt.

```mermaid
flowchart LR
    IN([text + attributes<br/>to hide])
    D[Defender<br/>rewrite + reasoning]
    A[Attacker<br/>guess attributes]
    L{Judge:<br/>leak?}
    U{Judge:<br/>utility ok?}
    R{rounds<br/>left?}
    P([PASS<br/>anonymized text])
    M([MAX_ITERS<br/>best candidate])

    IN --> D --> A --> L
    L -- no --> U
    U -- yes --> P
    L -- "yes · leak + reasons" --> R
    U -- "no · scores + reason" --> R
    R -. "yes · feedback → rewrite" .-> D
    R -- no --> M

    classDef agent fill:#e8f0ff,stroke:#3b82f6,color:#0b1220;
    classDef gate fill:#fff7e0,stroke:#d99e00,color:#0b1220;
    classDef good fill:#e7f8ee,stroke:#1f9d57,color:#0b1220;
    classDef warn fill:#fdeee0,stroke:#d9760b,color:#0b1220;
    class D,A agent
    class L,U,R gate
    class P good
    class M warn
```

> **Reading the diagram:** the Judge runs two gates in order — `utility ok?` is only checked when `leak?`
> says *no*. On a failure the Judge's findings are sent back to the Defender as **feedback** (the dotted
> edge): on a leak, *which attributes leaked and the reasons why* → rewrite harder; on low utility, *the
> scores + the reason + a "keep it safe" note* → rewrite lighter. The Defender always rewrites from the
> original text guided by this feedback, repeating until `max_iters`, then the best candidate so far is
> returned. Happy path: `input → Defender → Attacker → leak? no → utility ok? yes → PASS`.

### Components

- **NER/regex pre-scan** — detects direct identifiers (emails, phones, URLs, SSNs, credit cards, names) via regex and optionally spaCy NER. Findings are passed as advisory hints to the Defender prompt. Falls back to regex-only if the spaCy model is not installed.
- **Defender** — rewrites using abstraction / shifting / omission; guided by NER hints and targeted feedback each round.
- **Attacker** — chain-of-thought inference of each target attribute, with confidence + evidence spans.
- **Judge** — runs in two sequential stages:
  - **(1) Privacy gate** — decides whether each attribute is still inferable from the rewrite, reasoning over the Attacker's guesses and evidence. **A leak short-circuits the round** — utility is *not* scored and the Defender is sent back to rewrite harder.
  - **(2) Utility scoring** — reached only when nothing leaked, it scores `task_utility`, `factual_consistency`, `format_preserved`. Tracks the best candidate across rounds.
- **Ground truth validation** — when `ground_truth` is provided in the request, attacker guesses are validated against known true values using deterministic normalized exact/contains matching. This allows evaluation mode without relying solely on the Judge LLM's verdict.
- **Safe structured output** — all LLM structured output calls use a retry wrapper (`safe_structured_invoke`) that handles `None` returns and exceptions gracefully.

---

## Project layout

```
backend/
  config.yaml            # single config file (models, thresholds, weights)
  pyproject.toml         # dependencies (uv) + dev dependencies (pytest)
  uv.lock                # pinned lockfile
  app/
    main.py              # FastAPI; POST /api/anonymize, /api/anonymize_batch
    graph.py             # LangGraph wiring (compile)
    nodes.py             # defender / attacker / judge / finalize + router
    prompts.py           # the 3 agent prompts
    schemas.py           # Pydantic structured-output contracts
    scoring.py           # candidate ranking + retry-feedback builders
    llm.py               # LLM-agnostic factory + safe_structured_invoke
    state.py             # LangGraph shared state
    ner.py               # NER/regex pre-scan for direct identifiers
    matcher.py           # deterministic ground_truth validation
  tests/                 # pytest unit tests (65 tests, no LLM calls)
frontend/                # static UI (HTML/CSS/JS) + nginx reverse proxy
docker-compose.yml
CHANGES.txt              # detailed project status and change log
```

---

## API

### `POST /api/anonymize`

Streams `application/x-ndjson`, one JSON event per graph step (`start`, `node` ×N, `done` | `error`).

**Request body:**
```json
{
  "text": "I watched the moon landing with my dad when I was six.",
  "attributes_to_hide": ["age"],
  "utility_to_preserve": [],
  "channel": "text",
  "ground_truth": {"age": "6"}
}
```

- `ground_truth` (optional): A dict mapping attribute names to their true values. When provided, the Judge validates attacker guesses against these values using deterministic matching.

**Example:**
```bash
curl -N localhost:8080/api/anonymize -H 'Content-Type: application/json' \
  -d '{"text":"I watched the moon landing with my dad when I was six.","attributes_to_hide":["age"]}'
```

### `POST /api/anonymize_batch`

Processes multiple texts sequentially through the adversarial loop. Returns a single JSON response (not streaming).

**Request body:**
```json
{
  "texts": [
    "John Smith, age 42, works at Microsoft.",
    "Alice Jones, 30, is a doctor in Seattle."
  ],
  "attributes_to_hide": ["name", "age", "employer", "profession", "location"],
  "utility_to_preserve": [],
  "channel": "text",
  "ground_truth": [
    {"name": "John Smith", "age": "42", "employer": "Microsoft"},
    {"name": "Alice Jones", "age": "30", "profession": "doctor", "location": "Seattle"}
  ]
}
```

- Maximum batch size: 10 records
- `ground_truth` is a parallel array: `ground_truth[i]` applies to `texts[i]`

**Response:**
```json
{
  "batch_size": 2,
  "success_count": 2,
  "error_count": 0,
  "results": [
    {
      "index": 0,
      "status": "success",
      "original_text": "John Smith, age 42, works at Microsoft.",
      "final_text": "A professional in their forties works at a major tech company.",
      "verdict": "PASS",
      "rounds": 2,
      "leaked_attrs": [],
      "ground_truth_validation": {"name": {"matched": false}, "age": {"matched": false}}
    },
    ...
  ]
}
```

---

## Evaluation metrics

### Per-run metrics (current implementation)

The Judge LLM scores each anonymization run with these utility metrics:

- `task_utility` — how well the rewritten text preserves the intended task/purpose
- `informational_completeness` — whether key non-sensitive information is retained
- `factual_consistency` — whether the rewrite is factually consistent with the original
- `fluency` — readability and natural language quality
- `format_preserved` — whether structural formatting is maintained

Privacy verdict:
- `leaked` — boolean, whether any attribute was inferred by the Attacker
- `leaked_attrs` — list of attribute names that leaked
- `ground_truth_validation` — per-attribute match results when `ground_truth` is provided

### Dataset-level aggregation (future work)

A full evaluation harness would aggregate per-run metrics across a dataset:

- Average utility scores (task_utility, factual_consistency, etc.)
- Privacy success rate (% of records with no leaks)
- Leaked attribute rate (% of attributes leaked across all records)
- Average rounds needed to reach PASS
- Ground truth match rate (when ground_truth is provided)

This aggregation is not yet implemented.

---

## Known limitations

- **Semantic LLM matcher not implemented.** Ground truth validation uses deterministic string matching only (exact, contains, numeric extraction). A semantic LLM-based matcher for fuzzy matches is not yet implemented.
- **Cross-record re-identification not implemented.** Batch processing handles each record independently. Analysis of re-identification risk across multiple records in a dataset is not implemented.
- **Full evaluation harness not implemented.** There is no automated evaluation pipeline with ground truth datasets.
- **Frontend not re-tested.** The frontend has not been verified after recent backend changes (NER, ground_truth, batch).
- **Docker Vertex AI credentials.** Docker deployment with Vertex AI may require additional credential mounting that is not yet configured.
