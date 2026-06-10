"""Command-line utility to build FAISS vectorstores from data folders."""

from __future__ import annotations

from src.ingest import build_all_vectorstores


def main() -> None:
    """Build all configured vectorstores and print a short report."""
    results = build_all_vectorstores()
    for route, message in results.items():
        print(f"{route}: {message}")


if __name__ == "__main__":
    main()
