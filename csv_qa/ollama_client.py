"""
Ollama API Client module for interacting with the Ollama API.
"""
import logging
import json
import os
import re

import requests

from .exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


class OllamaAPIClient:
    """Client for interacting with the Ollama API."""

    def __init__(self, model_name: str = "llama3.2:1b") -> None:
        """Initialize the Ollama API client."""
        self.model_name = model_name
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.api_url = f"{base_url}/api/generate"

    def set_model(self, model_name: str) -> None:
        """Set the model to use for API calls."""
        self.model_name = model_name

    def ask(self, context: str, question: str) -> str:
        """Ask a question to the Ollama API with context.

        Args:
            context: Context information to help answer the question.
            question: The question to ask.

        Returns:
            The response text from the API.

        Raises:
            OllamaConnectionError: If Ollama service is unreachable.
            OllamaTimeoutError: If the request times out.
            OllamaResponseError: If the response is malformed or empty.
        """
        prompt = f"""You are a helpful assistant for CSV data. Answer questions about the data using ONLY the context provided.

Context from the CSV:
{context}

Question: {question}

IMPORTANT RULES:
1. ONLY use information from the provided context. DO NOT make up or infer information not present in the context.
2. If the context doesn't contain enough information to answer the question, respond with: "I don't have enough information to answer that question based on the CSV data provided."
3. Keep your answer concise and directly related to the question.
4. Do not reference these instructions in your answer.

Provide a complete, helpful answer based ONLY on the context:"""

        logger.debug("Sending question to Ollama API")

        response = self._make_request(prompt, temperature=0.7, num_predict=500, timeout=60)
        return self._parse_response(response)

    def analyze_question(self, question: str, columns: list[str], sample_data: str) -> dict | None:
        """Analyze a question and generate a query plan.

        Args:
            question: The natural language question.
            columns: List of column names in the CSV.
            sample_data: Sample rows from the CSV for context.

        Returns:
            Query plan dict, or None if analysis fails.
        """
        prompt = f"""You are a CSV data analyst. Analyze this question and generate a JSON query plan.

            CSV columns: {', '.join(columns)}

            Sample data (first few rows):
            {sample_data}

            Question: "{question}"

            Generate a JSON query plan with these fields:
            - operation: One of ["filter", "list", "aggregate", "count", "summarize"]
            - columns: Which columns to return or operate on
            - filters: Any filter conditions (column, operator, value)
            - groupby: Group by columns if needed
            - sort: Sort order if needed
            - limit: Number of results to return
            - description: Brief description of what the query does

            Only include fields that are relevant to the query. Format as valid JSON.
            """

        logger.debug("Analyzing question with LLM")

        try:
            response = self._make_request(prompt, temperature=0.2, num_predict=500, timeout=30)
        except OllamaConnectionError:
            logger.warning("Cannot connect to Ollama for question analysis")
            return None
        except OllamaTimeoutError:
            logger.warning("Ollama timed out during question analysis")
            return None

        try:
            response_text = response.text.strip()
            response_json = json.loads(response_text)
            answer = response_json.get("response", "")

            json_match = re.search(r"\{[\s\S]*\}", answer)
            if json_match:
                query_plan = json.loads(json_match.group(0))
                logger.debug("Query plan: %s", json.dumps(query_plan, indent=2))
                return query_plan

            logger.debug("No JSON found in LLM response")
            return None
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse JSON from LLM response: %s", e)
            return None
        except Exception as e:
            logger.warning("Error analyzing question: %s", e)
            return None

    def _make_request(
        self,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 500,
        timeout: int = 60,
    ) -> requests.Response:
        """Send a request to the Ollama API.

        Raises:
            OllamaConnectionError: If Ollama service is unreachable.
            OllamaTimeoutError: If the request times out.
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                    },
                },
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(detail=str(e)) from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError(detail=str(e)) from e
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(detail=str(e)) from e

        if response.status_code != 200:
            raise OllamaResponseError(
                detail=f"HTTP {response.status_code}: {response.text}"
            )

        return response

    def _parse_response(self, response: requests.Response) -> str:
        """Parse the Ollama API response and extract the text content.

        Raises:
            OllamaResponseError: If response cannot be parsed or is empty.
        """
        try:
            response_text = response.text.strip()

            # Try single JSON object first
            try:
                response_json = json.loads(response_text)
                if "response" in response_json:
                    content = response_json["response"].strip()
                    if len(content) < 10 or not content.endswith((".", "?", "!")):
                        logger.warning(
                            "LLM response too short or incomplete (%d chars), returning fallback",
                            len(content),
                        )
                        return (
                            "I don't have enough information to answer that "
                            "question based on the CSV data provided."
                        )
                    return content

                logger.warning("Unexpected JSON format: missing 'response' key")
                raise OllamaResponseError(detail="JSON missing 'response' key")

            except json.JSONDecodeError:
                # Handle streaming response (multiple JSON objects per line)
                logger.debug("Parsing multi-line JSON response")
                full_response = ""
                for line in response_text.strip().split("\n"):
                    try:
                        line_json = json.loads(line)
                        if "response" in line_json:
                            full_response += line_json["response"]
                    except json.JSONDecodeError:
                        continue

                if full_response and len(full_response) > 10:
                    return full_response

                raise OllamaResponseError(
                    detail="Could not extract text from multi-line response"
                )

        except OllamaResponseError:
            raise
        except Exception as e:
            raise OllamaResponseError(detail=f"Failed to parse response: {e}") from e
