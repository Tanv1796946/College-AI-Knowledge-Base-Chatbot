# College AI Knowledge Base – RAG-Based College Information Assistant

College AI Knowledge Base is an AI-powered college information assistant that allows students to ask questions about official college documents and receive relevant, document-grounded answers.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the college knowledge base before generating an answer.

---

## 1. Problem Statement

College information is often distributed across syllabus documents, academic calendars, fee documents, laboratory manuals, placement documents, and other official files.

Students may find it difficult and time-consuming to search through many documents to find specific information.

College AI Knowledge Base provides a single AI-powered interface where students can ask questions and receive answers based on the available official college documents.

---

## 2. Features

### Student Features

- Student login using name and email
- Ask questions about college information
- AI-generated answers based on college documents
- Conversation history
- New conversation support
- Source references for retrieved information
- Markdown-formatted answers
- Responsive user interface
- Loading and error handling

### RAG Features

- PDF and DOCX document processing
- Text extraction and chunking
- OCR support for scanned PDF pages
- Vector embeddings
- ChromaDB vector database
- Similarity-based document retrieval
- Google Gemini AI for answer generation
- Source document and page tracking
- Grounded responses to reduce hallucination

### Knowledge Base Categories

- College Information
- Syllabus
- Academic Calendar
- Laboratory Manuals
- Fees, Placement and Funding

---

## 3. Technology Stack

### Frontend

- React
- Vite
- JavaScript
- CSS
- React Markdown
- Remark GFM

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PyMuPDF
- python-docx

### AI / RAG

- Google Gemini
- ChromaDB
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)
- OCR for scanned documents

### Database

- SQLite
- ChromaDB

### Development Tools

- Git
- GitHub
- VS Code / Antigravity
- Uvicorn

---

## 4. Project Architecture

```text
College_AI_Knowledge_Base/
│
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── services/
│   │
│   ├── ingest.py
│   ├── test_api.py
│   ├── test_rag.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md

```text
College Documents
       ↓
Document Processing
       ↓
Text Extraction / OCR
       ↓
Text Chunking
       ↓
Vector Embeddings
       ↓
ChromaDB
       ↓
User Question
       ↓
Query Embedding
       ↓
Similarity Search
       ↓
Relevant Document Chunks
       ↓
Google Gemini
       ↓
Grounded Answer + Sources