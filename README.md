# EquitySearch 📈

An AI-powered financial news research assistant built with Python, Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Ollama.

## Overview

EquitySearch is a Retrieval-Augmented Generation (RAG) application that enables users to research financial news articles through natural-language questions.

Users can provide up to three financial news article URLs, process and index their content, and ask questions based on the indexed articles. The application uses a locally hosted Large Language Model (LLM) through Ollama to generate context-aware answers without relying on paid LLM APIs.

## Features

- Article Processing: Load and process up to three financial news articles using their URLs.
- Article Summarization: Generate concise summaries highlighting key events, companies, financial figures, and potential market impact.
- Semantic Search: Convert article content into embeddings and store them in a FAISS vector database.
- Research Mode: Ask questions across all indexed articles.
- Comparison Mode: Retrieve and compare relevant information from individual articles.
- Context-Aware Conversations: Handle follow-up questions using previous chat history.
- Source Display: View retrieved article excerpts and their source URLs.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Ollama (llama3.2:3b) |
| Framework | LangChain |
| Embeddings | Hugging Face (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Article Loading | UnstructuredURLLoader |

How It Works

1. URL Input: Users provide financial news article URLs.
2. Content Extraction: Article content is extracted and processed.
3. Text Embedding: The content is converted into numerical embeddings using Hugging Face.
4. Vector Indexing: Embeddings are stored in a FAISS vector database.
5. Information Retrieval: Relevant article chunks are retrieved based on user queries.
6. Answer Generation: Ollama generates context-aware responses using the retrieved information.

## Installation and Setup

### Prerequisites

- Python installed on your system.
- Ollama installed and running.
- Git (optional, for cloning the repository).

### 1. Clone the Repository

```bash
git clone <https://github.com/maiyagowri2845-cell/EquitySearch.git>
cd EquitySearch
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama

Download and install Ollama from https://ollama.com/.

Pull the required model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before launching the application.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## Acknowledgements

This project was inspired by the [Codebasics Rockybot project tutorial](https://www.youtube.com/watch?v=d4yCWBGFCEs&t=6639s).

The application was further customized to use locally hosted LLM inference with Ollama, Hugging Face embeddings, and FAISS, along with additional research and comparison capabilities.