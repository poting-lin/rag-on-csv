"""Unit tests for OllamaAPIClient configuration."""
import os
from unittest.mock import patch

from csv_qa.ollama_client import OllamaAPIClient


class TestOllamaClientConfig:
    """Tests for OllamaAPIClient URL configuration."""

    def test_default_base_url(self):
        """Default base URL should be localhost when no env var is set."""
        with patch.dict(os.environ, {}, clear=True):
            client = OllamaAPIClient()
            assert client.api_url == "http://localhost:11434/api/generate"

    def test_custom_base_url_from_env(self):
        """Base URL should be read from OLLAMA_BASE_URL env var."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://ollama:11434"}):
            client = OllamaAPIClient()
            assert client.api_url == "http://ollama:11434/api/generate"

    def test_env_var_trailing_slash_stripped(self):
        """Trailing slash in env var should be stripped."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://ollama:11434/"}):
            client = OllamaAPIClient()
            assert client.api_url == "http://ollama:11434/api/generate"
