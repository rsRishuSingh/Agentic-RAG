# RAG-Powered Financial Assistant

A Retrieval-Augmented Generation (RAG) agent specialized for financial queries, combining local document search (PDFs), web/Wikipedia lookup, and portfolio analytics. Built with **LangChain**, **ChromaDB**, **BM25**, and a GROQ-powered LLM (Qwen-3-32B), orchestrated via a stateful graph.

## RAG-Powered Financial Assistant
![WorkFLow](<Output and Workflow/graph.png>)

## Chainlit UI
![Output and Workflow/ui.png](<Output and Workflow/ui.png>)

## Output
![Output](<Output and Workflow/output.png>)
## Features

- **PDF Preprocessing & Chunking** - Extracts text and tables from financial PDFs. Splits content into semantic chunks (500-char with overlap) and stores in `all_docs.json`.
- **Hybrid Retrieval** - Combines BM25 and ChromaDB (vector search) ensembles for local PDF queries.
- **Web & Wiki Tools** - Real-time Google search via Serper API. Wikipedia lookups for contextual background.
- **Financial Analytics** - Calculates Sharpe Ratio, batting average, capture ratios, tracking error, and max drawdown.
- **Context Management** - Builds and compresses conversation history for LLM prompts. Timestamps and logs tool calls/responses.
- **Orchestration Graph** - Stateful routing of queries: intent classification → tool invocation → query refinement → final answer.
- **Optional Chainlit UI** - Interactive frontend for live demos and testing.

## Folder Structure

```
/ (project root)
├── __pycache__/            # Python cache
├── .chainlit/              # Chainlit state/config
├── .vscode/                # VS Code settings
├── chromaDB/               # Persisted ChromaDB store
├── myenv/                  # Virtual environment
├── PDFs/                   # Source PDFs (e.g. TESLA.pdf)
├── utils/                  # Helper notebooks (draw.ipynb)
├── Workflows/              # Workflow diagrams (graph.png)
├── .env                    # Environment variables
├── .gitignore              # Ignore rules
├── all_docs.json           # Serialized chunks & tables
├── app.py                  # Chainlit entrypoint
├── chainlit.md             # Chainlit docs
├── check_agent_log.json    # Tool call logs
├── main.py                 # Orchestration & tools
├── preprocessing.py        # PDF parsing & chunking
├── query.txt               # Sample queries
├── requirements.txt        # Python dependencies
├── utility.py              # Context & logging utilities
└── README.md               # Project overview
```

## Setup & Installation

Clone the repository:
```bash
git clone <repo_url>
cd <repo_dir>
```

Create & activate virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate    # macOS/Linux
myenv\Scripts\activate       # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure environment by copying `.env.example` → `.env` and setting required variables:
```ini
MODEL_NAME=qwen/qwen3-32b
SERPER_API_KEY=<your_serper_key>
ALPHAVANTAGE_API_KEY=<your_alpha_vantage_key>
PDF_DIR=PDFs/
ALL_DOCS_JSON=all_docs.json
CHROMA_DB_PATH=chromaDB/saved/
COLLECTION_NAME=RAG_DOCS
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## Running the Agent

**Preprocess PDFs** - Extract and chunk PDFs:
```bash
python preprocessing.py
```
This generates `all_docs.json` and populates the ChromaDB index.

**Start the Agent** via Chainlit UI:
```bash
chainlit run app.py
```

Or in CLI Mode:
```bash
python main.py
```

## Core Components

**PDF Preprocessing (`preprocessing.py`)** - `recursive_split` & `semantic_chunker` for semantic text chunking. Table extraction via `pdfplumber`. Serialization to `all_docs.json`.

**Context Utilities (`utility.py`)** - `get_context` & `compress_context` to build/compress chat history. `append_to_response` logs tool calls with IST timestamps. `remove_think` strips `<think>` blocks.

**Tools & Retrieval (`main.py`)** - Hybrid PDF Search using BM25 + ChromaDB. Web Tools including `google_search` and `wiki_lookup`. Financial Calculators for Sharpe, batting avg, capture ratios, tracking error, max drawdown.

**Orchestration Graph** - Built with `StateGraph` (langgraph). Nodes: Input → Intent Routing → ToolNodes → Check → Expand/Answer → END. Routers guide flow based on LLM/tool outputs.

## Customization

- **Add PDFs** - Place files in `PDFs/` & rerun preprocessing
- **Switch Models** - Update `MODEL_NAME` in `.env`
- **Tune Retrieval** - Adjust BM25 `k`, MMR `lambda_mult`, or ensemble weights
- **Extend Tools** - Decorate new functions with `@tool` and wire into `main.py`

## Contributing

Contributions are welcome! Open an issue or submit a pull request.

## License

Licensed under the MIT License.