# PDF-Based Question Answering with Pinecone & Groq

This project provides a simple Streamlit interface for uploading a PDF file, indexing its contents using Pinecone, and answering questions about the document by leveraging a Retrieval-Augmented Generation (RAG) approach using Groq's LLM API.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This application allows users to:
- **Upload a PDF file:** The application extracts the text content from the PDF using PyPDF2.
- **Index document content:** The text is embedded using a SentenceTransformer model and stored in a Pinecone vector store.
- **Query the document:** Users can ask questions related to the document content. The system retrieves the most relevant sections from the indexed data and uses Groq to generate an answer based on a custom prompt.

## Features

- **PDF Upload & Text Extraction:** Extracts text from multi-page PDFs.
- **Vector Indexing with Pinecone:** Uses Pinecone to store and query document embeddings.
- **Custom Embedding Model:** Leverages `SentenceTransformer` with the `"all-MiniLM-L6-v2"` model.
- **Retrieval-Augmented Generation:** Augments user queries with context from the PDF and generates responses using Groq's LLM.
- **Streamlit Interface:** A user-friendly interface for uploading PDFs and querying the indexed document.

## Prerequisites

- Python 3.7 or higher
- [Pinecone API Key](https://www.pinecone.io/start/)
- [Groq API Key](https://groq.com/) (or an equivalent access point for your LLM service)
- [Streamlit](https://streamlit.io/)
- Required Python packages (listed in [requirements.txt](requirements.txt) if available)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
