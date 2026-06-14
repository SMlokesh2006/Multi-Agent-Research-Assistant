# 🔬 Multi-Agent Research Assistant

A portfolio-grade multi-agent research assistant built with **LangGraph**, **Gemini Flash**, and **Tavily**. The system uses a Supervisor agent to orchestrate four specialized sub-agents through a shared state, producing comprehensive research reports with citations.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-Flash-orange?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## ✨ Features

### Core Architecture
- **Supervisor Agent** — Plans tasks and routes to sub-agents using tool-calling
- **Web Searcher** — Finds relevant URLs using Tavily search API
- **Content Reader** — Extracts and summarizes page content via Tavily Extract
- **Analyst** — Extracts key insights, identifies conflicts and knowledge gaps
- **Writer** — Produces structured markdown reports with inline citations

### Advanced Capabilities
- 🔄 **Parallel Execution** — LangGraph's Send API for concurrent agent work
- 👤 **Human-in-the-Loop** — Checkpoint-based pausing for user clarification
- 📡 **Real-time Streaming** — SSE-powered live updates to the web UI
- 💾 **Cross-session Memory** — SQLite persistence for research history
- 📊 **LangSmith Evaluation** — LLM-as-judge scoring for output quality
- ⚡ **Rate Limiting** — Token-bucket limiter for free tier API compliance
- 🗄️ **Response Caching** — SQLite cache to minimize redundant API calls

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│              (SSE Streaming + HITL API)              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 LangGraph StateGraph                  │
│                                                      │
│  START ──▶ Supervisor ──┬──▶ Web Searcher (×N) ──┐  │
│                │        ├──▶ Content Reader (×N) ─┤  │
│                │        ├──▶ Analyst ─────────────┤  │
│                │        ├──▶ Human Review ────────┤  │
│                │        └──▶ Writer ──▶ END       │  │
│                │                                  │  │
│                ◀──────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │          ResearchState (Shared State)           │  │
│  │  query | search_results | scraped_content |    │  │
│  │  analysis | report | messages | human_feedback │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Gemini  │ │  Tavily  │ │  SQLite  │
    │  Flash   │ │  Search  │ │  Store   │
    └──────────┘ └──────────┘ └──────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Gemini API key](https://aistudio.google.com/) (free tier works)
- [Tavily API key](https://tavily.com/) (1,000 free credits/month)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-research-assistant.git
cd multi-agent-research-assistant

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Run the Web UI

```bash
make dev
# Open http://localhost:8000 in your browser
```

### Run via CLI

```bash
make cli
# Or directly:
python -m src.cli "Compare RAG vs fine-tuning for domain-specific LLM applications"
```

---

## 📁 Project Structure

```
├── src/
│   ├── config.py              # Configuration & env vars
│   ├── state.py               # ResearchState TypedDict + data models
│   ├── graph.py               # LangGraph StateGraph construction
│   ├── persistence.py         # SQLite checkpointing & session memory
│   ├── server.py              # FastAPI streaming server
│   ├── cli.py                 # CLI interface
│   ├── evaluation.py          # LangSmith LLM-as-judge
│   ├── agents/
│   │   ├── supervisor.py      # Orchestrator with tool-calling
│   │   ├── web_searcher.py    # Tavily search agent
│   │   ├── content_reader.py  # Tavily Extract + fallback scraper
│   │   ├── analyst.py         # Insight extraction & conflict detection
│   │   └── writer.py          # Markdown report generator
│   └── utils/
│       ├── rate_limiter.py    # Token-bucket rate limiter
│       └── cache.py           # SQLite response cache with TTL
├── frontend/
│   ├── index.html             # Single-page application
│   ├── styles.css             # Dark mode + glassmorphism design
│   └── app.js                 # SSE streaming + HITL interaction
├── tests/
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## 🔧 Configuration

All configuration is done via environment variables or `.env` file:

| Variable | Required | Default | Description |
|:---|:---|:---|:---|
| `GOOGLE_API_KEY` | ✅ | — | Gemini API key |
| `TAVILY_API_KEY` | ✅ | — | Tavily API key |
| `LANGSMITH_API_KEY` | ❌ | — | Enables tracing & evaluation |
| `GEMINI_MODEL` | ❌ | `gemini-2.5-flash` | Model to use |
| `GEMINI_RPM_LIMIT` | ❌ | `15` | Requests per minute limit |
| `CACHE_TTL_HOURS` | ❌ | `24` | Search cache TTL |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/research` | Start a new research session |
| `GET` | `/research/{id}/stream` | SSE stream of agent updates |
| `POST` | `/research/{id}/feedback` | Submit HITL feedback |
| `GET` | `/research/{id}/status` | Check session status |
| `GET` | `/sessions` | List past research sessions |
| `GET` | `/health` | Health check |

---

## 🧪 Evaluation

Run the LangSmith evaluation suite (requires `LANGSMITH_API_KEY`):

```bash
make eval
```

Evaluates output on four criteria using LLM-as-judge:
- **Relevance** — Does the report answer the research query?
- **Accuracy** — Are citations properly used and verifiable?
- **Completeness** — Are all key aspects of the topic covered?
- **Coherence** — Is the report well-structured and readable?

---

## 💡 Design Decisions

| Decision | Rationale |
|:---|:---|
| **LangGraph over CrewAI** | Finer control over state, routing, and parallel execution |
| **Manual Supervisor** | Better prompt engineering control vs automated routing |
| **Send API for parallelism** | Dynamic fan-out based on runtime task count |
| **SQLite for persistence** | Zero-config, perfect for portfolio demos |
| **Tavily Extract + fallback** | Clean extraction with free-tier budget awareness |
| **Token-bucket rate limiter** | Stays within Gemini free tier RPM limits |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
