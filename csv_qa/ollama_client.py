"""
Ollama API Client module for interacting with the Ollama API
"""
import requests
import json
import traceback
import re


class OllamaAPIClient:
    """
    Client for interacting with the Ollama API
    """

    def __init__(self, model_name="llama3.2:1b", debug_mode=False):
        """Initialize the Ollama API client"""
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.api_url = "http://localhost:11434/api/generate"

    def set_model(self, model_name):
        """Set the model to use for API calls"""
        self.model_name = model_name

    def ask(self, context: str, question: str):
        """
        Ask a question to the Ollama API with context

        Args:
            context: Context information to help answer the question
            question: The question to ask

        Returns:
            str: The response from the API
        """
        try:
            # Create a dynamic prompt that includes the actual CSV column names
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

            if self.debug_mode:
                print("Asking Ollama API...")

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500  # Ensure we get a complete response
                    }
                },
                timeout=60  # Increased timeout
            )

            if response.status_code == 200:
                try:
                    response_text = response.text.strip()
                    try:
                        response_json = json.loads(response_text)
                        if "response" in response_json:
                            response_content = response_json["response"].strip(
                            )
                            # Check if response is too short or incomplete
                            if len(response_content) < 10 or not response_content.endswith((".", "?", "!")):
                                return "I don't have enough information to answer that question based on the CSV data provided."
                            return response_content
                        else:
                            if self.debug_mode:
                                print(
                                    f"Unexpected JSON response format: {response_json}")
                            return "I don't have enough information to answer that question based on the CSV data provided."
                    except json.JSONDecodeError:
                        # Handle case where response might be multiple JSON objects
                        if self.debug_mode:
                            print(
                                "Response contains multiple JSON objects, parsing line by line")
                        lines = response_text.strip().split('\n')
                        full_response = ""
                        for line in lines:
                            try:
                                line_json = json.loads(line)
                                if "response" in line_json:
                                    full_response += line_json["response"]
                            except json.JSONDecodeError:
                                continue

                        if full_response and len(full_response) > 10:
                            return full_response
                        return "I don't have enough information to answer that question based on the CSV data provided."
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error parsing JSON response: {e}")
                    return "I don't have enough information to answer that question based on the CSV data provided."
            else:
                return f"❌ Error from Ollama API: {response.status_code} - {response.text}"
        except requests.exceptions.ConnectionError:
            return "❌ Error: Could not connect to Ollama API. Make sure Ollama is running."
        except requests.exceptions.Timeout:
            return "❌ Error: Request to Ollama API timed out."
        except Exception as e:
            traceback.print_exc()
            return f"❌ Error: {str(e)}"

    def analyze_question(self, question, columns, sample_data):
        """
        Analyze a question and generate a query plan

        Args:
            question: The natural language question
            columns: List of column names in the CSV
            sample_data: Sample rows from the CSV for context

        Returns:
            dict: Query plan with operation type and parameters
        """
        try:
            # Format a prompt that asks the LLM to analyze the question and generate a query plan
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

            if self.debug_mode:
                print("Analyzing question with LLM...")

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 500
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                if self.debug_mode:
                    print(
                        f"Error: Failed to get response from Ollama API. Status code: {response.status_code}")
                return None

            try:
                response_text = response.text.strip()
                response_json = json.loads(response_text)
                answer = response_json.get('response', '')

                json_match = re.search(r'\{[\s\S]*\}', answer)
                if json_match:
                    json_str = json_match.group(0)
                    query_plan = json.loads(json_str)
                    if self.debug_mode:
                        print(
                            f"Query plan: {json.dumps(query_plan, indent=2)}")
                    return query_plan
                else:
                    if self.debug_mode:
                        print("No JSON found in LLM response")
                    return None
            except json.JSONDecodeError as e:
                if self.debug_mode:
                    print(f"Failed to parse JSON from LLM response: {e}")
                return None

        except Exception as e:
            if self.debug_mode:
                print(f"Error analyzing question: {str(e)}")
            return None
