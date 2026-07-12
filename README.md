
# Local AI Q&A Bot (Ollama + LangChain)

A local RAG (Retrieval-Augmented Generation) pipeline that answers questions from your PDF and text documents using Ollama and LangChain.

Here's what was created:

            Files       |                                          Purpose 
ingest.py  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Loads PDFs/text file wit embeds via Ollama nomic-embed-text stores in ChromaDB.  
rag_engine.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Sets up the RAG chain history, aware retrieval for follow-ups, context-aware QA prompt, supports model &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;switching.  
app.py&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Gradio web UI with chat, model dropdown, and clear button.  
requirements.txt&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;All dependencies.  
documents/ai_overview.txt&nbsp;&emsp;Sample document to test.  


## Features
- **Document Ingestion**: Load PDFs and text files, chunk them, embed with Ollama store in ChromaDB.
- **RAG Pipeline**: Retrieve relevant context and generate accurate answers using LangChain.
- **Chat History**: Supports follow-up questions with conversation memory.
- **Model Switching**: Compare outputs from different Ollama models (mistral, llama3, gemma, phi3).
- **Web UI**: Gradio-based chat interface running locally.

## Prerequisites
1. **Python 3.10+**
2. **Ollama** installed and running: https://ollama.com
3. Pull the required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## Setup

```bash
# 1. Clone / navigate to this directory
cd Godwind-Internship

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Run the project

```bash
# 1. Activate the virtual environment:
.\venv\Scripts\Activate.ps1

2. Install dependencies (if not already installed):
pip install -r requirements.txt

3. Run the app:
streamlit run app.py

The app will open in your browser at (http://localhost:8501),
You can then upload documents and ask questions.
```

## Result
<img width="1893" height="987" alt="Screenshot 2026-07-11 171830" src="https://github.com/user-attachments/assets/4d12bafc-eb9f-419d-a114-64c009d73cb7" />
## Usage

### 1. Add Documents
Place your `.pdf` or `.txt` files into the `documents/` folder or upload it in chatbot.

### 2. Ingest Documents
Run command to ingest documents
```bash
python ingest.py
```
This chunks your documents and builds the ChromaDB vector store in `chroma_db/`.

### 3. Launch the Chat Interface
```bash
python app.py
```
Open http://localhost:7860 in your browser.

### 4. Ask Questions
Type your question and get RAG-powered answers based uploaded document use the dropdown to switch between Ollama models.

## Project Structure

```
.
├── app.py              # Gradio web interface
├── ingest.py           # Document loading, chunking, embedding, ChromaDB storage
├── rag_engine.py       # RAG chain setup with Ollama + LangChain
├── requirements.txt    # Python dependencies
├── documents/          # Place your PDFs and text files here
├── chroma_db/          # Auto-generated ChromaDB vector store
└── README.md
```

## How It Works

1. **Ingestion** (`ingest.py`): Loads documents from `documents/` and splits them into chunks using `RecursiveCharacterTextSplitter`, embeds them with `nomic-embed-text` via Ollama and stores them in ChromaDB.

2. **RAG Engine** (`rag_engine.py`): On each query retrieves the top 4 relevant chunks, reformulates follow-up questions using chat history and generates an answer using the selected Ollama model.

3. **Interface** (`app.py`): Gradio web UI with a chatbot, model selector dropdown and clear button. Supports streaming conversation with history.

--------------------------------------------------------------------------------------------------------------------------
Created By: 
  [@Monesh Devadiga](https://github.com/Monesh-Devadiga)
