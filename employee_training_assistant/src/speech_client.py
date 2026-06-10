"""Speech-to-text support using Groq Whisper models."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import get_groq_api_key, get_groq_transcription_model, missing_api_key_message


@dataclass(frozen=True)
class TranscriptionResult:
    """Structured result for voice input transcription."""

    text: str
    status: str
    debug_error: str | None = None


def transcribe_audio(audio_bytes: bytes, file_name: str = "voice_question.wav") -> TranscriptionResult:
    """
    Transcribe recorded user speech with Groq Whisper.

    The app stays stable if the API key is missing, the model is unavailable, or
    the transcription request fails.
    """
    api_key = get_groq_api_key()
    if not api_key:
        return TranscriptionResult(
            text="",
            status="speech_missing_api_key",
            debug_error=missing_api_key_message(),
        )

    if not audio_bytes:
        return TranscriptionResult(
            text="",
            status="speech_empty_audio",
            debug_error="No audio bytes were received from the browser recorder.",
        )

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        transcription = client.audio.transcriptions.create(
            file=(file_name, audio_bytes),
            model=get_groq_transcription_model(),
            response_format="text",
        )
        text = _extract_transcription_text(transcription)
        if not text:
            return TranscriptionResult(
                text="",
                status="speech_no_text",
                debug_error="Groq returned an empty transcription.",
            )

        return TranscriptionResult(text=text, status="speech_success")
    except Exception as exc:
        return TranscriptionResult(
            text="",
            status="speech_error",
            debug_error=f"{exc.__class__.__name__}: {exc}",
        )


def _extract_transcription_text(transcription: object) -> str:
    """Normalize Groq transcription responses across text/object formats."""
    if isinstance(transcription, str):
        return transcription.strip()

    text = getattr(transcription, "text", "")
    return str(text).strip()
