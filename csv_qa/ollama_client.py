import logging
import json
import os
import random
import re
import time
from collections.abc import Callable, Iterator

import requests

from csv_qa.exceptions import (
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 0.5

FALLBACK_ANSWER = (
    "I don't have enough information to answer that question based on the CSV data provided."
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def estimate_confidence(answer: str, context: str) -> float:
    """Score how well an LLM answer is grounded in the provided context.

    Returns a value in [0.0, 1.0]. The score is heuristic — it rewards
    overlap of content words and numbers between answer and context,
    penalises very short or fallback answers. Useful as a post-hoc
    signal to flag possible hallucinations, not a ground-truth measure.
    """
    if not answer:
        return 0.0
    if answer.strip() == FALLBACK_ANSWER:
        return 0.3

    stripped = answer.strip()
    if len(stripped) < 15:
        return 0.2

    context_lower = context.lower()
    answer_words = {w.lower() for w in _WORD_RE.findall(stripped)}
    if not answer_words:
        return 0.4

    grounded_words = sum(1 for w in answer_words if w in context_lower)
    word_ratio = grounded_words / len(answer_words)

    answer_numbers = set(_NUMBER_RE.findall(stripped))
    if answer_numbers:
        grounded_numbers = sum(1 for n in answer_numbers if n in context)
        number_ratio = grounded_numbers / len(answer_numbers)
    else:
        number_ratio = 1.0

    score = 0.4 + 0.4 * word_ratio + 0.2 * number_ratio
    return min(1.0, max(0.0, score))


def _with_retry(
    fn: Callable[[], requests.Response],
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> requests.Response:
    """Run fn with exponential backoff on transient failures.

    Retries on transport errors (connection, timeout) and on 5xx/429
    HTTP responses. Returns the Response on success or on non-retriable
    4xx so callers can inspect the status and produce their own errors.
    Raises OllamaConnectionError / OllamaTimeoutError only after all
    transport attempts are exhausted.
    """
    last_transport_exc: Exception | None = None
    last_response: requests.Response | None = None

    for attempt in range(attempts):
        should_retry = False
        try:
            response = fn()
        except requests.exceptions.Timeout as e:
            last_transport_exc = e
            should_retry = True
            if attempt == attempts - 1:
                raise OllamaTimeoutError(detail=str(e)) from e
        except requests.exceptions.ConnectionError as e:
            last_transport_exc = e
            should_retry = True
            if attempt == attempts - 1:
                raise OllamaConnectionError(detail=str(e)) from e
        except requests.exceptions.RequestException as e:
            last_transport_exc = e
            should_retry = True
            if attempt == attempts - 1:
                raise OllamaConnectionError(detail=str(e)) from e
        else:
            last_response = response
            if response.status_code == 429 or response.status_code >= 500:
                should_retry = attempt < attempts - 1
                if not should_retry:
                    return response
            else:
                return response

        if should_retry:
            sleep_for = base_delay * (2**attempt) + random.uniform(0, base_delay)
            logger.warning(
                "Ollama call failed (attempt %d/%d): %s. Retrying in %.2fs",
                attempt + 1,
                attempts,
                last_transport_exc or (last_response and last_response.status_code),
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_response is not None:
        return last_response
    raise OllamaConnectionError(detail=f"Exhausted retries: {last_transport_exc}")


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
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.api_url = f"{base_url}/api/generate"

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
        answer = self._parse_response(response)
        confidence = estimate_confidence(answer, context)
        if confidence < 0.5:
            logger.warning("Low-confidence answer (score=%.2f): %s", confidence, answer[:120])
        return answer

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
        """Send a request to the Ollama API, with retry on transient failures."""
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

        response = _with_retry(lambda: requests.post(self.api_url, json=payload, timeout=timeout))
        if response.status_code != 200:
            raise OllamaResponseError(detail=f"HTTP {response.status_code}: {response.text}")
        return response

    def ask_stream(
        self,
        context: str,
        question: str,
        temperature: float = 0.2,
        num_predict: int = 500,
        timeout: int = 60,
    ) -> Iterator[str]:
        """Stream a grounded answer token-by-token.

        Yields text chunks as they arrive from Ollama. Consumers join or
        display them incrementally. Errors after the first chunk end the
        stream; errors before the first chunk raise.
        """
        prompt = (
            f"{QA_SYSTEM_RULES}\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{question}\n\n"
            f"## Answer\n"
        )
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }

        def _post() -> requests.Response:
            return requests.post(self.api_url, json=payload, timeout=timeout, stream=True)

        response = _with_retry(_post)
        if response.status_code != 200:
            response.close()
            raise OllamaResponseError(detail=f"HTTP {response.status_code}: {response.text}")

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
        finally:
            response.close()

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
                        return FALLBACK_ANSWER
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

                raise OllamaResponseError(detail="Could not extract text from multi-line response")

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
