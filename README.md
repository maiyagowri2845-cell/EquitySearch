# EquitySearch 📈

**An AI-powered financial news research assistant built with Python, Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Ollama.**

## Overview

EquitySearch is a Retrieval-Augmented Generation (RAG) application that helps users research financial news articles through natural-language questions.

Users can provide up to three financial news article URLs, process and index their content, and ask questions based on the indexed articles. The application uses a locally running LLM through Ollama to generate context-aware answers.

## Features

- **Article Processing:** Load and process up to three financial news articles using their URLs.
- **Article Summarization:** Generate concise summaries highlighting key events, companies, financial figures, and potential market impact.
- **Semantic Search:** Convert article content into embeddings and store them in a FAISS vector database.
- **Research Mode:** Ask questions across all indexed articles.
- **Comparison Mode:** Retrieve and compare relevant information from individual articles.
- **Context-Aware Conversations:** Handle follow-up questions using previous chat history.
- **Source Display:** View the retrieved article excerpts and their source URLs.

## Technologies Used

- **Language:** Python
- **Frontend:** Streamlit
- **LLM:** Ollama (`llama3.2:3b`)
- **Framework:** LangChain
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **Article Loading:** UnstructuredURLLoader

## Acknowledgements

This project was inspired by the [Codebasics Rockybot project tutorial](https://www.youtube.com/watch?v=d4yCWBGFCEs&t=6639s).

The application was further customized to use locally hosted LLM inference with Ollama, Hugging Face embeddings, and FAISS.