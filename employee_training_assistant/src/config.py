"""Central configuration for the AI Training Assistant."""

from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "AI Employee Training & Policy Assistant"
APP_SUBTITLE = "RAG-Based Onboarding Chatbot with Decision-Node Routing"

GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 4

ASSISTANT_AVATAR = ":material/smart_toy:"
USER_AVATAR = ":material/person:"

ROUTE_COMPANY_OVERVIEW = "company_overview"
ROUTE_ROLE_DOCUMENTS = "role_documents"
ROUTE_ADMIN_POLICIES = "admin_policies"
ROUTE_DIRECT_LLM = "direct_llm"

ROUTE_LABELS = {
    ROUTE_COMPANY_OVERVIEW: "Company Overview Knowledge Base",
    ROUTE_ROLE_DOCUMENTS: "Role and Team Documents",
    ROUTE_ADMIN_POLICIES: "Policy and Administrative Documents",
    ROUTE_DIRECT_LLM: "Direct LLM Response",
}

VECTORSTORE_PATHS = {
    ROUTE_COMPANY_OVERVIEW: BASE_DIR / "vectorstores" / "company_overview",
    ROUTE_ROLE_DOCUMENTS: BASE_DIR / "vectorstores" / "role_documents",
    ROUTE_ADMIN_POLICIES: BASE_DIR / "vectorstores" / "admin_policies",
}

DATA_PATHS = {
    ROUTE_COMPANY_OVERVIEW: BASE_DIR / "data" / "company_overview",
    ROUTE_ROLE_DOCUMENTS: BASE_DIR / "data" / "role_documents",
    ROUTE_ADMIN_POLICIES: BASE_DIR / "data" / "admin_policies",
}


def get_groq_api_key() -> str | None:
    """Return the Groq API key from the local environment."""
    return os.getenv("GROQ_API_KEY")


def get_groq_model_name() -> str:
    """Return the configured Groq chat model."""
    return os.getenv("GROQ_MODEL", GROQ_MODEL_NAME)


def get_groq_transcription_model() -> str:
    """Return the configured Groq speech-to-text model."""
    return os.getenv("GROQ_TRANSCRIPTION_MODEL", GROQ_TRANSCRIPTION_MODEL)


def missing_api_key_message(variable_name: str = "GROQ_API_KEY") -> str:
    """Return a clear message for missing local credentials."""
    return (
        f"Missing {variable_name}. Create a local .env file from .env.example "
        "and add your key. Do not commit .env to GitHub."
    )
