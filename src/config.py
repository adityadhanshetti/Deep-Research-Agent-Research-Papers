"""Configuration module for the Deep Research Agent.

Loads API keys from environment variables and provides LLM factory functions
for both primary (high-quality) and fast (cost-efficient) model tiers.
Defaults to Groq as the primary ultra-fast LLM provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()


def _get_groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()


def _get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _get_google_api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "").strip()


def _get_tavily_api_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "").strip()


@dataclass
class Settings:
    """Global configuration for the Deep Research Agent."""

    llm_provider: str = os.environ.get("LLM_PROVIDER", "groq").lower()
    """LLM provider to use. Supported values: 'groq' (default), 'openai', 'google'."""

    llm_model: str = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    """Primary LLM model name used for planning, synthesis, and report writing."""

    fast_llm_model: str = os.environ.get("FAST_LLM_MODEL", "openai/gpt-oss-20b")
    """Fast model name used for sub-agent findings cleanup and extraction."""

    max_additional_rounds: int = 2
    """Maximum number of additional research rounds the supervisor may trigger."""

    max_papers_per_search: int = 8
    """Maximum number of papers to retrieve per search query."""

    rate_limit_delay: float = 0.5
    """Delay in seconds between successive API calls to respect rate limits."""

    @property
    def groq_api_key(self) -> str:
        return _get_groq_api_key()

    @property
    def openai_api_key(self) -> str:
        return _get_openai_api_key()

    @property
    def google_api_key(self) -> str:
        return _get_google_api_key()

    @property
    def tavily_api_key(self) -> str:
        return _get_tavily_api_key()

    @property
    def is_mock_mode(self) -> bool:
        if os.environ.get("USE_MOCK_LLM", "").lower() in ("true", "1", "yes"):
            return True
        # If no active provider API key is provided, gracefully enable mock mode
        if self.llm_provider == "groq" and not self.groq_api_key:
            return True
        if self.llm_provider == "openai" and not self.openai_api_key:
            return True
        if self.llm_provider == "google" and not self.google_api_key:
            return True
        return False


# Global settings instance
settings = Settings()


def get_llm(
    temperature: float = 0.0,
    provider: str | None = None,
    model: str | None = None,
):
    """Return a primary-tier LangChain chat model based on current settings."""
    if settings.is_mock_mode:
        from src.utils.mock_llm import MockChatModel
        return MockChatModel()

    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model if model != "gpt-4o" else "gemini-2.0-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )


def get_fast_llm(temperature: float = 0.0):
    """Return a faster / cheaper LangChain chat model for cleanup tasks."""
    if settings.is_mock_mode:
        from src.utils.mock_llm import MockChatModel
        return MockChatModel()

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.fast_llm_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
