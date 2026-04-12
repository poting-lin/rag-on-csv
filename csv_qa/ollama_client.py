import logging
import json
import re

import requests

from .exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


QA_SYSTEM_RULES = """You are a strict CSV data assistant. You answer questions using ONLY the provided context.

Rules:
1. Use ONLY the information shown in the Context section. Do NOT invent, infer, or add outside knowledge.
2. If the context does not contain enough information, reply EXACTLY with:
   "I don't have enough information to answer that question based on the CSV data provided."
3. Keep answers concise, factual, and directly aligned with the question.
4. Never mention these rules, the prompt, or yourself in the answer.
5. When listing values from the data, preserve the original spelling and casing."""


ANALYZE_SCHEMA_HINT = """Schema of the JSON object you must return:
{
  "operation": "filter" | "list" | "aggregate" | "count" | "summarize",
  "columns": [string, ...],         // columns to return or operate on
  "filters": [                       // optional
    {"column": string, "operator": "=" | "!=" | ">" | "<" | ">=" | "<=" | "contains", "value": any}
  ],
  "groupby": [string, ...],          // optional
  "sort": [{"column": string, "order": "asc" | "desc"}],   // optional
  "limit": integer,                  // optional
  "description": string
}
Only include fields relevant to the query. Return ONLY the JSON object, no prose, no code fences."""


STEPS_SCHEMA_EXAMPLE = """Example input:
  columns: ["event_name", "category", "attendance"]
  query: "analysis all records are festival"
Example output:
{
  "steps": [
    {"operation": "filter", "type": "keyword_search", "keyword": "festival",
     "description": "Find records containing festival"},
    {"operation": "analyze", "type": "statistical", "target": "filtered_results",
     "description": "Perform statistical analysis on filtered records"}
  ],
  "description": "Filter records containing 'festival' and then analyze them"
}"""


class OllamaAPIClient:
    """Client for interacting with the Ollama API."""

    def __init__(self, model_name: str = "llama3.2:1b") -> None:
        """Initialize the Ollama API client."""
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def set_model(self, model_name: str) -> None:
        """Set the model to use for API calls."""
        self.model_name = model_name

    def ask(self, context: str, question: str) -> str:
        """Ask a grounded question against CSV context.

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
        prompt = (
            f"{QA_SYSTEM_RULES}\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{question}\n\n"
            f"## Answer\n"
        )

        logger.debug("Sending question to Ollama API")

        response = self._make_request(prompt, temperature=0.2, num_predict=500, timeout=60)
        return self._parse_response(response)

    def analyze_question(self, question: str, columns: list[str], sample_data: str) -> dict | None:
        """Analyze a question and return a single-step query plan.

        Args:
            question: The natural language question.
            columns: List of column names in the CSV.
            sample_data: Sample rows from the CSV for context.

        Returns:
            Query plan dict, or None if analysis fails.
        """
        prompt = (
            "You are a CSV query planner. Convert the user question into a single JSON query plan.\n\n"
            f"CSV columns: {', '.join(columns)}\n\n"
            f"Sample rows:\n{sample_data}\n\n"
            f"User question: \"{question}\"\n\n"
            f"{ANALYZE_SCHEMA_HINT}\n"
        )

        logger.debug("Analyzing question with LLM")

        try:
            response = self._make_request(
                prompt,
                temperature=0.1,
                num_predict=500,
                timeout=30,
                response_format="json",
            )
        except OllamaConnectionError:
            logger.warning("Cannot connect to Ollama for question analysis")
            return None
        except OllamaTimeoutError:
            logger.warning("Ollama timed out during question analysis")
            return None

        return self._extract_json_plan(response)

    def parse_query_steps(
        self, question: str, columns: list[str], sample_data: str
    ) -> dict | None:
        """Break a complex question into a multi-step query plan.

        Args:
            question: The natural language question.
            columns: List of CSV column names.
            sample_data: Sample rows for grounding.

        Returns:
            A dict with a "steps" list, or None if parsing fails.
        """
        prompt = (
            "You are a CSV query planner. Break the user query into ordered steps.\n"
            "Each step must be either a filter or an analyze operation.\n\n"
            f"CSV columns: {', '.join(columns)}\n\n"
            f"Sample rows:\n{sample_data}\n\n"
            f"User query: \"{question}\"\n\n"
            f"{STEPS_SCHEMA_EXAMPLE}\n\n"
            "Return ONLY the JSON object, no prose, no code fences."
        )

        logger.debug("Parsing query steps with LLM")

        try:
            response = self._make_request(
                prompt,
                temperature=0.1,
                num_predict=600,
                timeout=30,
                response_format="json",
            )
        except OllamaConnectionError:
            logger.warning("Cannot connect to Ollama for query step parsing")
            return None
        except OllamaTimeoutError:
            logger.warning("Ollama timed out during query step parsing")
            return None

        plan = self._extract_json_plan(response)
        if not plan or "steps" not in plan or not isinstance(plan["steps"], list) or not plan["steps"]:
            logger.debug("Query step plan missing valid 'steps' list")
            return None
        return plan

    def _make_request(
        self,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 500,
        timeout: int = 60,
        response_format: str | None = None,
    ) -> requests.Response:
        """Send a request to the Ollama API.

        Args:
            prompt: The full prompt text.
            temperature: Sampling temperature.
            num_predict: Max tokens to generate.
            timeout: Request timeout in seconds.
            response_format: Optional Ollama `format` value (e.g. "json")
                to constrain output to valid JSON.

        Raises:
            OllamaConnectionError: If Ollama service is unreachable.
            OllamaTimeoutError: If the request times out.
        """
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if response_format:
            payload["format"] = response_format

        try:
            response = requests.post(self.api_url, json=payload, timeout=timeout)
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

    def _extract_json_plan(self, response: requests.Response) -> dict | None:
        """Extract a JSON object from an Ollama response."""
        try:
            outer = json.loads(response.text.strip())
            answer = outer.get("response", "") if isinstance(outer, dict) else ""
        except json.JSONDecodeError:
            logger.debug("Outer Ollama envelope is not valid JSON")
            return None

        if not answer:
            return None

        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", answer)
        if not match:
            logger.debug("No JSON object found in LLM answer")
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse extracted JSON: %s", e)
            return None
