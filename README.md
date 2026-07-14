# Strafgesetzbuch RAG Assistant

A Retrieval-Augmented Generation (RAG) application for querying the German Criminal Code (StGB).

The project uses a custom document ingestion pipeline to parse legal HTML documents, create embeddings, store them in a Qdrant vector database, and generate answers based only on retrieved legal sections.

## Features

- HTML legal document parsing
- Semantic search using Gemini embeddings
- Vector storage with Qdrant
- Context-based answer generation with Gemini
- Source-aware responses with legal references
- Environment-based API key management
- Basic protection against prompt injection attempts

## Tech Stack

- Python
- Google Gemini API
- Qdrant Vector Database
- BeautifulSoup
- Retrieval-Augmented Generation (RAG)

## Project Structure

```
strafgesetzbuch_rag/
│
├── parser.py              # Legal document parsing and vector database creation
├── rag.py                 # Retrieval and answer generation
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .env                   # API key (not included in repository)
└── qdrant_local_db/       # Local vector database (not included in repository)
```

## Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

First, process the legal document and create the vector database:

```bash
python parser.py
```

Then start the RAG assistant:

```bash
python rag.py
```

## Architecture

The application follows a simple RAG pipeline:

1. Legal HTML documents are parsed into structured sections.
2. Text sections are converted into embeddings using Gemini.
3. Embeddings are stored in Qdrant.
4. User questions are converted into embeddings.
5. Relevant legal sections are retrieved.
6. Gemini generates an answer based only on the retrieved context.

## Notes

This project was created as a practical exploration of Retrieval-Augmented Generation systems applied to legal documents.

The goal was to build a complete RAG workflow without relying on external orchestration frameworks, using direct SDK integrations with the underlying services.
