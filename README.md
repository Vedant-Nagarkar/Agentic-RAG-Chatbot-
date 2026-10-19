# Enterprise IT Support Agentic RAG Chatbot

An internal IT support assistant that answers employee questions from a company's **private knowledge base first**, checks whether the evidence is good enough, and only goes to the public web when the question is not company-specific.

Built with **LangGraph, FastAPI, Pinecone, OpenAI, Tavily, SQLite, and a plain HTML/CSS/JavaScript frontend**.

> **Status:** working prototype with a sample knowledge base and an evaluation harness. It is not production-ready. See [Known limitations](#known-limitations).

---

## 1. The problem

**Customer (fictional):** NovaRetail, a 3,000-employee retail company.

The IT team keeps many documents: VPN steps, password rules, MFA policy, software rules, laptop troubleshooting and service-desk runbooks. Employees still open repeat tickets because they cannot find the right document, keyword search returns too much, and a normal chatbot may invent an answer.

Two example questions show the design goal:

| Question | Right behavior |
|---|---|
| "How do I connect to the company VPN?" | Answer from the private KB. No web search. |
| "What is the latest Microsoft Teams outage guidance?" | This is public vendor information. The KB is weak, so use web search and label the answer as external. |

A third case matters just as much:

| Question | Right behavior |
|---|---|
| "What is the process for requesting a company-issued mobile phone?" | This is company-specific and not in the KB. Say so. **Do not** answer it with generic internet advice. |

---

## 2. What it does

- Routes each message: small talk, or an IT question.
- Decides whether a question is **company-specific** (web search not allowed) or **public** (web search allowed).
- Retrieves from a private Pinecone knowledge base and grades the evidence.
- If the evidence is weak, **rewrites the query and retries the KB once** before considering the web.
- Falls back to web search only for public questions. Internal details are masked before the query leaves the system.
- Generates answers only from the retrieved evidence, and shows the decision path (trace) in the UI.
- Lets an admin upload PDF, TXT, MD or DOCX files into the knowledge base.
- Writes each question, source used and trace to a SQLite audit log.

---

## 3. Workflow

```text
Question
   │
[1] Route ── small talk ───────────────────────────────► Direct answer
   │         (also decides: company-specific or public?)
   ▼
[2] Retrieve from private KB (Pinecone)
   ▼
[3] Grade evidence
   ├── GOOD ───────────────────────────────────────────► Answer from KB
   └── WEAK
        │
        ├─ retry left? ─► [4] Rewrite query ─► back to [2]
        │
        └─ no retry left
             ├─ company-specific ───────────────────────► Stop: "not in the KB"
             └─ public
                  ▼
               [5] Web search (query masked first)
                  ▼
               [6] Grade web evidence
                  ├── GOOD ───────────────────────────► Answer from web (labeled external)
                  └── WEAK ───────────────────────────► Stop: insufficient evidence
```

### Why this order

- **Retry the KB before the web.** A weak KB result is often just a badly worded question. Every web call sends text outside the company, so it comes last.
- **Company-specific questions never go to the web.** A public answer to "how do I use our backup VPN gateway?" would be wrong for the employee. When the router is unsure, it treats the question as company-specific.
- **The model controls what happens next**, based on the state so far. This is what makes it "agentic" and not a fixed retrieve-then-generate chain.

---

## 4. Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent workflow | LangGraph | Stateful routing and conditional steps |
| LLM | OpenAI `gpt-4o-mini` (configurable) | Routing, grading, query rewriting, answers |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensions) | Vectorizing documents and questions |
| Vector database | Pinecone (serverless) | Private knowledge base |
| Web search | Tavily | Fallback for public questions |
| API | FastAPI | REST endpoints and page serving |
| Frontend | HTML, CSS, JavaScript | Chat UI with a live agent trace |
| Audit log | SQLite | Question, source and trace per request |
| Packaging | Docker | Container image |

---

## 5. Project structure

```text
.
├── app/
│   ├── api/routes.py            # /api/chat, /api/ingest, /api/health
│   ├── core/
│   │   ├── config.py            # Settings read from .env
│   │   └── logging.py
│   ├── rag/
│   │   ├── state.py             # Graph state and structured LLM outputs
│   │   ├── vectorstore.py       # Pinecone index, embeddings, stable chunk IDs
│   │   └── workflow.py          # The full LangGraph workflow
│   ├── services/
│   │   ├── audit.py             # SQLite audit log
│   │   └── ingestion.py         # PDF/TXT/MD/DOCX loading and chunking
│   └── main.py                  # FastAPI app
├── data/sample_kb/              # Sample NovaRetail documents
├── evals/
│   ├── run_eval.py              # Evaluation runner
│   ├── cases.json               # 29 labeled questions
│   └── injection_cases.json     # Prompt-injection checks (separate namespace)
├── static/                      # CSS and JavaScript
├── templates/index.html
├── ingest_sample_kb.py          # Loads the sample KB into Pinecone
├── run.py                       # Local dev server
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 6. Setup

**Requirements:** Python 3.11 or newer, and accounts and API keys for OpenAI, Pinecone and Tavily.

### 1. Create a virtual environment and install packages

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Open `.env` and fill in `OPENAI_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, and a real `ADMIN_API_KEY`. Never commit `.env`.

### 3. Load the sample knowledge base

```bash
python ingest_sample_kb.py
```

The first request creates the Pinecone index if it does not exist.

### 4. Run the app

```bash
python run.py
```

Open **http://127.0.0.1:8080**. API docs are at **http://127.0.0.1:8080/docs**.

---

## 7. Configuration reference

All settings come from `.env` (or real environment variables).

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | none (required) | OpenAI key |
| `TAVILY_API_KEY` | none (required) | Tavily key |
| `PINECONE_API_KEY` | none (required) | Pinecone key |
| `PINECONE_INDEX_NAME` | `agentic-rag-chatbot` | Pinecone index |
| `PINECONE_NAMESPACE` | `company-it-kb` | Namespace inside the index |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K` | `4` | Chunks retrieved per search |
| `MAX_RETRIES` | `1` | Query rewrites allowed before the web step. `0` skips the rewrite. |
| `ADMIN_API_KEY` | `change-me` | Key required for uploads |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `REDACT_TERMS` | `NovaRetail,NovaSecure` | Names masked as `[company]` before web search |
| `APP_ENV` | `development` | When not `development`, uploads are refused while the admin key still starts with `change-me` |
| `AUDIT_DB_PATH` | `data/audit.db` | SQLite audit file |
| `UPLOAD_DIR` | `uploads/` | Where uploaded files are saved |

> **Changing `EMBEDDING_MODEL` after you have indexed documents** changes the vector size. The app will return an error that explains the mismatch. It never deletes an index by itself. Use a new `PINECONE_INDEX_NAME` and re-ingest.

---

## 8. Try it

Example questions and the path each one is designed to take:

| Ask | Expected path |
|---|---|
| How do I connect to the company VPN? | Route → KB, retrieve, grade GOOD, answer from KB |
| What is the process for requesting a company-issued mobile phone? | KB weak, rewrite, KB weak, **stop**. No web call. |
| What is the latest Microsoft Teams outage guidance? | KB weak, rewrite, KB weak, web search, answer labeled external |
| Hello! | Direct answer. No retrieval. |

The right-hand panel shows every step the agent took for the last question, including the exact query sent to the web.

---

## 9. Adding documents

Use **+ Add Company Document** in the sidebar, enter the admin key, and choose a `.pdf`, `.txt`, `.md` or `.docx` file.

- The key is compared in constant time. Files over `MAX_UPLOAD_MB`, empty files and unreadable files are rejected with a clear message.
- Each chunk gets a **stable ID** built from the file name and chunk position. Uploading the same file again **overwrites** its chunks instead of creating duplicates.
- A file with the same name replaces the earlier one.
- If you edit a file so it becomes shorter, chunks from the old longer version can remain. See [Known limitations](#known-limitations).

---

## 10. API

### `GET /api/health`

Returns `{"status": "ok", "service": "..."}`.

### `POST /api/chat`

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I connect to the company VPN?"}'
```

Response:

```json
{
  "answer": "...",
  "source_used": "private_kb",
  "trace": [
    "Router → KB (company-specific: True)",
    "Private KB retrieval → 4 chunks",
    "KB evidence grade → GOOD",
    "Answer generation → PRIVATE KB"
  ],
  "citations": [{"title": "company_it_handbook.md", "url": "", "type": "private_kb"}],
  "rewritten_query": "How do I connect to the company VPN?"
}
```

`source_used` is one of `private_kb`, `web_search`, `direct` or `insufficient_evidence`. Question length is 2 to 3000 characters. Internal errors return a generic 500 message. The full error is written to the server log.

### `POST /api/ingest`

Requires the `X-Admin-Key` header. Send the file as multipart form data.

```bash
curl -X POST http://127.0.0.1:8080/api/ingest \
  -H "X-Admin-Key: YOUR_ADMIN_KEY" \
  -F "file=@policy.md"
```

| Status | Meaning |
|---|---|
| 200 | Indexed. Returns file name and chunk count. |
| 400 | Unsupported type, empty file, or unreadable file |
| 401 | Wrong admin key |
| 413 | File larger than `MAX_UPLOAD_MB` |
| 503 | Server not in development and admin key not changed |

---

## 11. Security and privacy

### What is in place

- **Private first, web last.** Company-specific questions never trigger a web search.
- **Query masking before web search.** Email addresses, IPv4 addresses, internal host names (`.local`, `.corp`, `.internal`, `.lan`), ticket IDs (INC, TKT, REQ, CHG, SR), key-like strings and the names in `REDACT_TERMS` are replaced with placeholders. The exact text sent is shown in the trace.
- **Prompt-injection defense.** Retrieved documents and web results are wrapped in `<untrusted_data>` tags with an instruction to treat them as reference material only. Fake closing tags inside documents are removed. The app has no tools that take actions, so the worst case is a wrong answer, not a changed system.
- **Index safety.** A vector-size mismatch stops the app with an error and never deletes the index.
- **Upload hardening.** Constant-time key check, size limit, empty and unreadable file handling, and no default admin key outside development.
- **Frontend safety.** All text is HTML-escaped before display. Source links are only made clickable for `http` and `https` addresses.
- **No secrets in git.** `.env`, uploaded files and the audit database are git-ignored.

### What is not in place

- Masking uses pattern matching. It will **miss** personal names, project code names and details written in ordinary sentences.
- There is **one shared admin key**, and there is no end-user login.
- There is **no rate limiting** on `/api/chat`.
- The audit log stores the **raw question text**, with no user ID and no retention policy.
- Prompt-injection defense lowers the risk but cannot remove it. It is checked with `evals/injection_cases.json`, which is a small test, not a proof.

---

## 12. Evaluation

The `evals/` folder contains a labeled test set and a runner. It calls the real workflow and checks each question automatically.

```bash
python -m evals.run_eval                # 29 questions against the sample KB
python -m evals.run_eval --runs 3       # repeat to find flaky cases
```

For the injection checks, point the app at a **separate namespace** that holds the poisoned test documents, and run:

```bash
python -m evals.run_eval --cases evals/injection_cases.json
```

**What it measures**

- Overall pass rate (a question passes only if every check passes)
- Router accuracy: company-specific vs public
- Source accuracy: KB, web, direct or stop
- **Privacy violations**: a company question reaching the web, or internal text found in a web query. This must be 0.
- Required and forbidden text in answers
- Latency (average, p90, maximum)


**Latest results** (36 questions, 1 run, gpt-4o-mini, sample KB)

| Metric | Result |
|---|---|
| Overall pass rate | 35/36 |
| Router accuracy (company vs public) | 17/17 |
| Source accuracy | 35/36 |
| Privacy violations | 0 |
| Latency (avg / p90 / max) | 5.7 / 11.6 / 12.5 s |

**How to read these numbers.** There are only 29 questions, so one case moves the score by about 3 points. The answer checks look for specific facts, not for full answer quality. The labels are the author's judgement of correct behavior. Treat the results as a regression check, not as a benchmark.

---

## 13. Docker

```bash
docker build -t it-support-rag .
docker run --rm -p 8080:8080 --env-file .env \
  -e AUDIT_DB_PATH=/app/storage/audit.db \
  -e UPLOAD_DIR=/app/storage/uploads \
  -v it_support_storage:/app/storage \
  it-support-rag
```

The named volume keeps the audit log and uploaded files across restarts. The container currently runs as root and has no health check.

---

## 14. Known limitations

- **The router and the grader are language-model judgements.** They can be wrong, and results vary between runs. The evaluation set exists to measure this.
- **Vague questions are not clarified.** "My work app is acting strange" ends in "not found" when it should ask which app.
- **No conversation memory.** Each question is answered alone, so follow-ups like "what about on Mac?" do not work.
- **Retrieval is vector-only.** There is no keyword search or reranking, which hurts exact terms such as error codes.
- **Citations are file-level.** They name the file, not the section or page.
- **Shorter re-uploads leave stale chunks.** Deleting old chunks by ID prefix is not implemented.
- **No streaming.** Each answer needs several model calls and arrives all at once.
- **Small test coverage.** There is an evaluation harness but no unit-test suite.
- **No deployed demo yet.**

## 15. Roadmap

1. Run and publish the evaluation, then fix failures by cause: bug, wrong label, or flaky.
2. Ask a clarifying question for vague requests.
3. Add conversation history.
4. Add hybrid search and a reranker, then re-run the evaluation to measure the change.
5. Add section-level citations.
6. Add unit tests and a CI workflow.
7. Add per-user login and rate limiting.
8. Deploy a public demo.

---

## License

MIT. See [LICENSE](LICENSE).