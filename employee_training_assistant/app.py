"""Streamlit chatbot UI for the AI Training Assistant."""

from __future__ import annotations

import hashlib
import json

import streamlit as st
import streamlit.components.v1 as components

from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    ASSISTANT_AVATAR,
    ROUTE_LABELS,
    USER_AVATAR,
    VECTORSTORE_PATHS,
)
from src.ingest import build_all_vectorstores
from src.response_generator import generate_answer
from src.speech_client import transcribe_audio


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=ASSISTANT_AVATAR,
    layout="wide",
)

st.markdown(
    """
    <style>
    .assistant-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .assistant-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.24);
    }
    .assistant-title h1 {
        margin: 0;
        padding: 0;
    }
    .assistant-title p {
        margin: 0.15rem 0 0;
        color: #5f6b7a;
    }
    .voice-panel {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.7rem 0.85rem;
        border: 1px solid rgba(15, 118, 110, 0.18);
        border-radius: 12px;
        background: rgba(15, 118, 110, 0.06);
        margin: 0.5rem 0 0.85rem;
    }
    .voice-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        animation: voicePulse 1.5s infinite ease-in-out;
    }
    .voice-panel strong {
        display: block;
    }
    .voice-panel span {
        display: block;
        color: #5f6b7a;
        font-size: 0.88rem;
    }
    @keyframes voicePulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.28); }
        50% { transform: scale(1.04); box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Create Streamlit session keys used by the chat UI."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am your AI Training Assistant. Ask me about the company, "
                    "your role, admin policies, or general learning topics."
                ),
            }
        ]
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False
    if "voice_rate" not in st.session_state:
        st.session_state.voice_rate = 0.95
    if "voice_pitch" not in st.session_state:
        st.session_state.voice_pitch = 1.0
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None


def render_sidebar() -> None:
    """Render sidebar information about routing paths."""
    with st.sidebar:
        st.header("Routing Paths")
        st.markdown(
            """
**Company overview**  
Mission, values, departments, working hours, company background.

**Role documents**  
Role expectations, team responsibilities, trainer duties, reporting structure.

**Admin policies**  
Leave, expenses, reimbursement, payroll, attendance, IT policy, security.

**Direct LLM**  
General learning or unrelated questions that do not need company documents.
"""
        )
        st.divider()
        st.header("Voice Avatar")
        st.toggle("Speak assistant answers", key="voice_enabled")
        st.slider("Voice speed", 0.7, 1.25, key="voice_rate", step=0.05)
        st.slider("Voice pitch", 0.8, 1.2, key="voice_pitch", step=0.05)
        st.caption("Speak mode uses Groq Whisper for speech input and your browser for speech output.")
        st.divider()
        render_knowledge_base_status()
        if st.button("Build knowledge base", use_container_width=True):
            with st.spinner("Reading PDFs and building FAISS vectorstores..."):
                results = build_all_vectorstores()

            st.success("Knowledge base build completed.")
            for route, message in results.items():
                st.caption(f"{ROUTE_LABELS.get(route, route)}: {message}")


def render_knowledge_base_status() -> None:
    """Show whether the FAISS indexes needed for RAG are available."""
    st.header("Knowledge Base")
    all_ready = True
    for route, vectorstore_path in VECTORSTORE_PATHS.items():
        is_ready = (vectorstore_path / "index.faiss").exists() and (vectorstore_path / "index.pkl").exists()
        all_ready = all_ready and is_ready
        icon = "Ready" if is_ready else "Missing"
        st.caption(f"{ROUTE_LABELS.get(route, route)}: {icon}")

    if all_ready:
        st.success("RAG retrieval is enabled.")
    else:
        st.warning("Build FAISS indexes to enable document-based answers.")


def render_chat_history() -> None:
    """Display prior chat messages."""
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("route"):
                st.caption(f"Detected route: {ROUTE_LABELS.get(message['route'], message['route'])}")
            if message.get("routing_reason"):
                st.caption(message["routing_reason"])
            if message.get("ANSWER_MODE"):
                st.caption(f"Answer mode: {message['ANSWER_MODE']}")
            if message.get("retrieved_context"):
                with st.expander("Retrieved context"):
                    for index, chunk in enumerate(message["retrieved_context"], start=1):
                        st.markdown(f"**Context {index}**")
                        st.write(chunk)
            if message.get("EVALUATION_SCORES"):
                render_evaluation_scores(message["EVALUATION_SCORES"])
            if message.get("llm_debug_error"):
                with st.expander("Debug"):
                    st.caption(f"LLM status: {message.get('LLM_STATUS')}")
                    st.code(message["llm_debug_error"])


def render_voice_avatar(answer: str) -> None:
    """Render the speaking avatar and ask the browser to speak the latest answer."""
    if not st.session_state.voice_enabled or not answer.strip():
        return

    speech_text = _clean_text_for_speech(answer)
    payload = json.dumps(
        {
            "text": speech_text,
            "rate": st.session_state.voice_rate,
            "pitch": st.session_state.voice_pitch,
        }
    )
    components.html(
        f"""
        <style>
        .voice-panel {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.7rem 0.85rem;
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 12px;
            background: rgba(15, 118, 110, 0.06);
            font-family: "Source Sans Pro", Arial, sans-serif;
        }}
        .voice-avatar {{
            width: 42px;
            height: 42px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: linear-gradient(135deg, #0f766e, #2563eb);
            animation: voicePulse 1.5s infinite ease-in-out;
        }}
        .voice-panel strong {{
            display: block;
            color: #1f2937;
            font-size: 0.95rem;
        }}
        .voice-panel span {{
            display: block;
            color: #5f6b7a;
            font-size: 0.84rem;
        }}
        @keyframes voicePulse {{
            0%, 100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.28); }}
            50% {{ transform: scale(1.04); box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }}
        }}
        </style>
        <div class="voice-panel">
            <div class="voice-avatar">{ASSISTANT_AVATAR}</div>
            <div>
                <strong>Speaking answer</strong>
                <span>The assistant avatar is reading the latest response aloud.</span>
            </div>
        </div>
        <script>
        const payload = {payload};
        if ("speechSynthesis" in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(payload.text);
            utterance.rate = payload.rate;
            utterance.pitch = payload.pitch;
            utterance.lang = "en-IN";
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=95,
    )


def _clean_text_for_speech(text: str) -> str:
    """Remove markdown/source clutter before sending text to browser speech."""
    cleaned = text.replace("**", "").replace("`", "")
    cleaned = cleaned.replace("Source:", "According to source")
    return " ".join(cleaned.split())


def render_evaluation_scores(scores: dict) -> None:
    """Display BLEU and ROUGE document-overlap scores."""
    with st.expander("BLEU and ROUGE scores"):
        if not scores.get("available"):
            st.caption(scores.get("reason", "Scores are not available for this answer."))
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BLEU", f"{scores['bleu']:.4f}")
        col2.metric("ROUGE-1", f"{scores['rouge1']:.4f}")
        col3.metric("ROUGE-2", f"{scores['rouge2']:.4f}")
        col4.metric("ROUGE-L", f"{scores['rougeL']:.4f}")
        st.caption(scores.get("note", "Scores are calculated against retrieved document context."))


def handle_user_prompt(prompt: str) -> None:
    """Process a user prompt and append both user and assistant messages."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Routing query, retrieving context, and generating answer..."):
            result = generate_answer(prompt)

        route_label = ROUTE_LABELS.get(result["route"], result["route"])
        st.caption(f"Detected route: {route_label}")
        st.caption(result["routing_reason"])
        st.caption(f"Answer mode: {result['ANSWER_MODE']}")
        st.caption(f"LLM status: {result['LLM_STATUS']}")
        st.markdown(result["final_answer"])
        render_voice_avatar(result["final_answer"])

        if result["retrieved_context"]:
            with st.expander("Retrieved context"):
                for index, chunk in enumerate(result["retrieved_context"], start=1):
                    st.markdown(f"**Context {index}**")
                    st.write(chunk)

        render_evaluation_scores(result["EVALUATION_SCORES"])

        if result["llm_debug_error"]:
            with st.expander("Debug"):
                st.caption(f"LLM status: {result['LLM_STATUS']}")
                st.code(result["llm_debug_error"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["final_answer"],
            "route": result["route"],
            "routing_reason": result["routing_reason"],
            "retrieved_context": result["retrieved_context"],
            "LLM_STATUS": result["LLM_STATUS"],
            "ANSWER_MODE": result["ANSWER_MODE"],
            "EVALUATION_SCORES": result["EVALUATION_SCORES"],
            "llm_debug_error": result["llm_debug_error"],
        }
    )


def render_voice_input() -> None:
    """Record a spoken question, transcribe it, and send it through the chatbot."""
    st.markdown("#### Voice Question")
    audio_file = st.audio_input("Record your question, then wait for transcription")
    if not audio_file:
        return

    audio_bytes = audio_file.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash == st.session_state.last_audio_hash:
        return

    st.session_state.last_audio_hash = audio_hash
    with st.spinner("Transcribing your speech with Groq Whisper..."):
        transcription = transcribe_audio(audio_bytes, audio_file.name)

    if transcription.status != "speech_success":
        st.warning("I could not understand the recorded question.")
        with st.expander("Speech debug"):
            st.caption(f"Speech status: {transcription.status}")
            st.code(transcription.debug_error or "No debug details available.")
        return

    st.caption(f"Transcribed question: {transcription.text}")
    handle_user_prompt(transcription.text)


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()
    render_sidebar()

    st.markdown(
        f"""
        <div class="assistant-header">
            <div class="assistant-avatar">{ASSISTANT_AVATAR}</div>
            <div class="assistant-title">
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_chat_history()
    render_voice_input()

    prompt = st.chat_input("Ask an employee training question...")
    if prompt:
        handle_user_prompt(prompt)


if __name__ == "__main__":
    main()
