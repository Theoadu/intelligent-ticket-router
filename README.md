# Intelligent Ticket Router

An observable, multi-agent RAG (Retrieval-Augmented Generation) system for intelligent customer support ticket routing. The system automatically classifies incoming support queries and routes them to specialized domain agents (HR, Tech, Finance) that provide contextual answers using relevant documentation.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)

## 🎯 Overview

The Intelligent Ticket Router is designed to handle customer support queries by:
1. **Classifying** incoming queries into departments (HR, Tech, Finance, or Other)
2. **Routing** queries to specialized agents with domain-specific knowledge
3. **Retrieving** relevant information from vector databases for each domain
4. **Generating** contextual, accurate responses using LLMs
5. **Evaluating** response quality with automated scoring
6. **Tracing** all operations for observability using Langfuse

## 🏗 Architecture

The system follows a multi-agent orchestration pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator                            │
│  (LLM-based Query Classification)                       │
│  - Classifies query into: HR / Tech / Finance / Other   │
│  - Returns confidence score and reasoning               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ HR Agent │  │Tech Agent│  │Finance   │
│          │  │          │  │Agent     │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ HR Docs  │  │Tech Docs │  │Finance   │
│ Vector   │  │ Vector   │  │Docs      │
│   DB     │  │   DB     │  │Vector DB │
└──────────┘  └──────────┘  └──────────┘
```

### Core Components

1. **Orchestrator** (`src/orchestrator.py`)
   - LLM-powered query classifier
   - Routes queries to appropriate domain agents
   - Fallback classification using keyword matching
   - Integrates with Langfuse for tracing

2. **Domain Agents** (`src/agents/`)
   - **HR Agent**: Handles employee-related queries (leave, benefits, policies)
   - **Tech Agent**: Handles technical support queries (IT issues, deployments, access)
   - **Finance Agent**: Handles financial queries (expenses, payroll, budgets)
   - Each agent uses `ConversationalRetrievalChain` with domain-specific retrievers

3. **Document Pipeline** (`src/document_pipeline.py`, `src/ingest.py`)
   - Loads documents from `data/<domain>_docs/` directories
   - Splits documents into chunks using `RecursiveCharacterTextSplitter`
   - Generates embeddings using OpenAI's `text-embedding-3-small`
   - Stores in ChromaDB vector databases (one per domain)
   - Synthesizes additional variants if needed for testing

4. **Evaluator** (`src/evaluator.py`)
   - Automated quality assessment of generated responses
   - Scores responses from 1-10 based on accuracy, completeness, and relevance
   - Provides rationale and identifies issues
   - Uses structured output with Pydantic models

5. **Observability** (Langfuse Integration)
   - Traces all LLM calls and agent executions
   - Logs evaluation scores
   - Enables debugging and performance monitoring

## 🛠 Technology Stack

### Core Frameworks
- **LangChain** (v1.1.0+): Orchestration framework for LLM applications
  - `langchain-openai`: OpenAI integration
  - `langchain-chroma`: ChromaDB vector store integration
  - `langchain-community`: Community integrations
  - `langchain-text-splitters`: Document chunking utilities

### Vector Database
- **ChromaDB** (v1.3.5+): Persistent vector storage for embeddings

### LLM & Embeddings
- **OpenAI API**:
  - **Model**: `gpt-4o-mini` (default completion model)
  - **Embeddings**: `text-embedding-3-small` (default embedding model)

### Observability
- **Langfuse** (v3.10.1+): LLM observability and tracing platform

### Additional Libraries
- **python-dotenv**: Environment variable management
- **rich**: Enhanced CLI output formatting
- **tiktoken**: Token counting for OpenAI models
- **Pydantic**: Data validation and structured outputs

### Development
- **uv**: Fast Python package installer and runner
- **Python**: 3.13+

## ✨ Features

- **Intelligent Query Classification**: LLM-based classification with fallback keyword matching
- **Domain-Specific Knowledge Retrieval**: Separate vector databases per department
- **Conversational Context**: Maintains chat history for follow-up questions
- **Automated Quality Evaluation**: Built-in response scoring system
- **Full Observability**: End-to-end tracing with Langfuse
- **Persistent Storage**: ChromaDB for efficient vector search
- **Flexible Document Ingestion**: Supports .txt, .md, .csv, .pdf files
- **Test Suite**: Predefined test queries with expected classifications

## 📁 Project Structure

```
intelligent-ticket-router/
├── data/                           # Knowledge base documents
│   ├── hr_docs/                   # HR policies and guides
│   ├── tech_docs/                 # Technical support documentation
│   └── finance_docs/              # Finance procedures and policies
├── src/
│   ├── agents/                    # Domain-specific agents
│   │   ├── base_agent.py         # Base agent utilities and retriever creation
│   │   ├── hr_agent.py           # HR support agent
│   │   ├── tech_agent.py         # Technical support agent
│   │   ├── finance_agent.py      # Finance support agent
│   │   └── orchestrator.py       # Query classifier and router (alternate)
│   ├── ingest.py                 # Batch ingestion with synthesis
│   ├── orchestrator.py           # Main orchestrator implementation
│   ├── run_system.py             # CLI runner with test execution
│   ├── evaluator.py              # Response quality evaluator
├── main.py                        # Main entry point for queries
├── test_queries.json             # Test queries with expected classifications
├── run_results.json              # Test execution results
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Locked dependencies
├── .env                          # Environment variables (not in repo)
├── .env.example                  # Example environment configuration
└── README.md                     # This file
```

## 📋 Prerequisites

- Python 3.13 or higher
- OpenAI API key
- (Optional) Langfuse account for observability

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd intelligent-ticket-router
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

## ⚙️ Configuration

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys**
   ```bash
   # OpenAI (Required)
   OPENAI_API_KEY=your-openai-api-key

   # Langfuse (Optional - for tracing)
   LANGFUSE_PUBLIC_KEY=pk-lf-xxx
   LANGFUSE_SECRET_KEY=sk-lf-xxx
   LANGFUSE_HOST=https://cloud.langfuse.com

   # Chroma persistence path
   CHROMA_PERSIST_DIR=./.chromadb

   # Optional: Customize models
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   OPENAI_COMPLETION_MODEL=gpt-4o-mini
   ```

## 💻 Usage

### 1. Ingest Documents and Build Vector Databases

Before running queries, you need to ingest the documents and create vector databases:

```bash
uv run src/run_system.py --ingest --build --run-tests
```

This will:
- Load documents from `data/hr_docs/`, `data/tech_docs/`, and `data/finance_docs/`
- Split them into chunks
- Generate embeddings
- Store in ChromaDB (`.chromadb/` directory)


Output includes:
- The query
- Generated answer
- Quality score (1-10)
- Langfuse trace URL (if configured)

### 3. Run Test Suite

Execute all test queries from `test_queries.json`:

```bash
uv run src/run_system.py --run-tests --langfuse
```

Results are saved to `run_results.json` with:
- Query
- Classification (HR/Tech/Finance)
- Answer
- Evaluation score and rationale

### 4. Add Custom Documents

Add your own documents to the appropriate directory:
- HR documents → `data/hr_docs/`
- Tech documents → `data/tech_docs/`
- Finance documents → `data/finance_docs/`

Then re-run ingestion:
```bash
uv run src/run_system.py --ingest
```

## 🔍 How It Works

### Query Flow

1. **User submits a query** via `main.py` or `run_system.py`

2. **Orchestrator classifies the query**
   - Uses GPT-4o-mini to analyze the query
   - Returns classification (HR/Tech/Finance/Other) with confidence score
   - Falls back to keyword matching if LLM classification fails

3. **Query is routed to the appropriate agent**
   - HR Agent for employee-related queries
   - Tech Agent for technical issues
   - Finance Agent for financial matters

4. **Agent retrieves relevant context**
   - Queries the domain-specific ChromaDB vector database
   - Retrieves top-k most relevant document chunks (k=6)
   - Uses semantic similarity via embeddings

5. **Agent generates response**
   - Uses `ConversationalRetrievalChain` with GPT-4o-mini
   - Combines retrieved context with the query
   - Generates a contextual, accurate answer
   - Returns source documents used

6. **Response is evaluated**
   - Evaluator scores the response (1-10)
   - Checks accuracy, completeness, and relevance
   - Provides rationale and identifies issues

7. **All operations are traced**
   - Langfuse captures every LLM call
   - Traces include inputs, outputs, latency, and token usage
   - Evaluation scores are logged for analysis

### Vector Database Structure

Each domain has its own ChromaDB collection:
- `hr_collection`: HR documents and policies
- `tech_collection`: Technical support guides
- `finance_collection`: Financial procedures

Documents are:
- Chunked into ~800 character segments with 150 character overlap
- Embedded using OpenAI's `text-embedding-3-small` model
- Stored with metadata (domain, chunk_index)
- Persisted to disk in `.chromadb/` directory

## 📊 Evaluation

The system includes an automated evaluator that scores responses on a 1-10 scale:

- **10**: Perfect (accurate, complete, relevant, proper citations)
- **8-9**: Excellent with minor room for improvement
- **6-7**: Good but missing some details or context
- **4-5**: Acceptable but has noticeable issues
- **1-3**: Poor or hallucinated

Evaluation criteria:
- **Accuracy**: Answer matches the query intent
- **Completeness**: All aspects of the query are addressed
- **Relevance**: Information is pertinent to the question
- **Source Usage**: Proper citation of retrieved documents

Scores are automatically logged to Langfuse for tracking model performance over time.

## 🔗 Observability with Langfuse

If Langfuse is configured, you get:
- **Trace visualization**: See the complete execution flow
- **Token usage tracking**: Monitor API costs
- **Latency metrics**: Identify bottlenecks
- **Quality scores**: Track evaluation results
- **Debugging**: Inspect inputs/outputs at each step

Access traces at: `https://cloud.langfuse.com/trace/{trace_id}`

## 🤝 Contributing

To extend the system:

1. **Add a new domain**:
   - Create `data/<domain>_docs/` directory
   - Add agent class in `src/agents/<domain>_agent.py`
   - Update orchestrator classification logic
   - Update `ingest.py` to include new domain

2. **Customize chunking strategy**:
   - Modify `chunk_size` and `chunk_overlap` in `ingest.py`
   - Adjust `k` parameter for retrieval in agent classes

3. **Use different models**:
   - Update `OPENAI_COMPLETION_MODEL` in `.env`
   - Update `OPENAI_EMBEDDING_MODEL` in `.env`

## 📝 License



## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) for LLM orchestration
- [OpenAI](https://openai.com/) for language models and embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Langfuse](https://langfuse.com/) for observability
