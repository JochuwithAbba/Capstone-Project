# AI Employee Training & Policy Assistant

## RAG-Based Onboarding Chatbot with Decision-Node Routing

## Project Overview

This capstone project is a Streamlit-based AI assistant for employee onboarding, training, and policy questions. It routes each user question to the right decision path, retrieves relevant company document context from FAISS vectorstores when needed, and generates a clear final answer using an LLM API.

The app is designed to demonstrate practical AI engineering patterns: retrieval-augmented generation, decision-node routing, modular Python code, safe API key handling, document ingestion, and transparent response metadata.

## Key Features

- Streamlit chatbot interface for employee questions.
- Decision-node routing for company overview, role documents, admin policies, or direct LLM answers.
- FAISS vector search over PDF and text documents.
- Groq LLM response generation with safe fallback behavior.
- Groq Whisper speech-to-text for voice questions.
- Browser text-to-speech for assistant answers.
- BLEU and ROUGE document-overlap scoring for retrieved-context answers.
- Clear route, retrieval, answer mode, and LLM status display in the UI.

## Architecture

```text
User question
-> Optional voice transcription
-> Query router
-> Route decision
-> FAISS retrieval when a document route is selected
-> Prompt template
-> Groq LLM client
-> Final response
-> Optional evaluation scores and speech output
```

Core modules:

- `app.py`: Streamlit UI and user interaction flow.
- `src/router.py`: decision-node routing logic.
- `src/retriever.py`: FAISS/vectorstore retrieval.
- `src/llm_client.py`: LLM API calls and fallback handling.
- `src/prompts.py`: prompt templates.
- `src/response_generator.py`: orchestration for route, retrieval, prompt, and answer.
- `src/config.py`: environment variables, constants, models, and paths.
- `src/utils.py`: shared helpers.
- `src/ingest.py`: document loading and FAISS vectorstore creation.
- `src/speech_client.py`: speech-to-text support.
- `src/evaluation.py`: BLEU and ROUGE scoring.

## Tech Stack

- Python
- Streamlit
- Groq API
- Groq Whisper
- LangChain
- FAISS
- Sentence Transformers
- PyPDF
- python-dotenv
- NLTK
- rouge-score

## Folder Structure

```text
employee_training_assistant/
|-- app.py
|-- build_vectorstores.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .env.example
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- router.py
|   |-- retriever.py
|   |-- llm_client.py
|   |-- prompts.py
|   |-- response_generator.py
|   |-- utils.py
|   |-- ingest.py
|   |-- speech_client.py
|   |-- evaluation.py
|
|-- data/
|   |-- company_overview/
|   |-- role_documents/
|   |-- admin_policies/
|
|-- vectorstores/
|   |-- company_overview/
|   |-- role_documents/
|   |-- admin_policies/
|
|-- screenshots/
|-- docs/
|-- backup_original_project/
```

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` in the project root and add your local API key:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
```

Only `.env.example` should be committed. The real `.env` file is ignored by Git.

## Build the Knowledge Base

Run this after adding or changing documents in `data/`:

```bash
python build_vectorstores.py
```

You can also build the knowledge base from the Streamlit sidebar.

## How to Run Locally

From the project root:

```bash
streamlit run app.py
```

## Demo Scenarios

- Ask: "What is the company mission?"
- Ask: "What are the responsibilities of a new trainer?"
- Ask: "How many leaves can an employee take?"
- Ask: "What is the reimbursement process?"
- Ask: "What should I do if I forget my laptop password?"
- Ask: "Can you explain what RAG means?"

## Capstone Deliverables

- Streamlit chatbot application.
- Modular Python source code.
- Company, role, and admin policy document folders.
- FAISS vectorstore build script.
- README and setup documentation.
- Screenshot folder for demo evidence.
- Docs folder for report, PPT, and submission materials.

## Security Note

API keys and secrets are not committed to this repository. The app loads credentials from local environment variables using `python-dotenv`. Keep real keys in `.env` or your local shell environment only.
