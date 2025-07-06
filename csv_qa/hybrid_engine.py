"""
Hybrid Engine - combines structured queries, semantic search, and LLM analysis
"""
import pandas as pd
from typing import Dict, Any, Optional
from .structured_query_engine import StructuredQueryEngine
from .enhanced_vector_search import CSVAwareVectorSearch
from .ollama_client import OllamaAPIClient


class HybridCSVEngine:
    """Combines multiple approaches for comprehensive CSV analysis"""

    def __init__(self, debug_mode=False, model_name="llama3.2:1b"):
        self.debug_mode = debug_mode
        self.structured_engine = StructuredQueryEngine(debug_mode=debug_mode)
        self.vector_engine = CSVAwareVectorSearch(debug_mode=debug_mode)
        self.llm_client = OllamaAPIClient(
            model_name=model_name, debug_mode=debug_mode)

    def answer_question(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Use multiple approaches to answer the question

        Args:
            question: The user's question
            df: The pandas DataFrame to analyze

        Returns:
            Dict with answer, confidence, method used, and additional data
        """
        if self.debug_mode:
            print(f"Hybrid engine processing: {question}")

        # 1. Try structured approach first (fastest and most accurate for structured queries)
        structured_result = self._try_structured_query(question, df)
        if structured_result['success'] and structured_result.get('confidence', 0) > 0.8:
            if self.debug_mode:
                print("Structured query succeeded with high confidence")
            return {
                'answer': structured_result['result'],
                'confidence': structured_result.get('confidence', 0.9),
                'method': 'structured',
                'data': structured_result.get('data'),
                'success': True
            }

        # 2. Try semantic analysis with enhanced vector search
        semantic_result = self._try_semantic_analysis(question, df)
        if semantic_result['success']:
            if self.debug_mode:
                print("Semantic analysis succeeded")

            # If we also had a partial structured result, combine insights
            if structured_result['success']:
                combined_answer = self._combine_results(
                    structured_result, semantic_result)
                return {
                    'answer': combined_answer,
                    'confidence': 0.85,
                    'method': 'hybrid',
                    'data': structured_result.get('data'),
                    'success': True
                }

            return {
                'answer': semantic_result['result'],
                'confidence': semantic_result.get('confidence', 0.7),
                'method': 'semantic',
                'data': semantic_result.get('data'),
                'success': True
            }

        # 3. Fall back to LLM with basic context
        llm_result = self._try_llm_analysis(question, df)
        return {
            'answer': llm_result['result'],
            'confidence': llm_result.get('confidence', 0.6),
            'method': 'llm',
            'data': llm_result.get('data'),
            'success': llm_result['success']
        }

    def _try_structured_query(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Attempt direct DataFrame operations"""
        try:
            result = self.structured_engine.execute_query(question, df)

            if result['success']:
                # Calculate confidence based on result quality
                confidence = self._calculate_structured_confidence(
                    question, result)
                result['confidence'] = confidence

                if self.debug_mode:
                    print(
                        f"Structured query result (confidence: {confidence:.2f}): {result['result'][:100]}...")

            return result

        except Exception as e:
            if self.debug_mode:
                print(f"Structured query failed: {e}")
            return {'success': False, 'error': str(e)}

    def _try_semantic_analysis(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Attempt semantic analysis using enhanced vector search"""
        try:
            # Create structured chunks
            chunks = self.vector_engine.create_structured_chunks(df)

            if not chunks:
                return {'success': False, 'error': 'No chunks created'}

            # Build vector index
            self.vector_engine.build_vector_index(chunks)

            # Retrieve relevant context
            context = self.vector_engine.retrieve_context(question, top_k=5)

            if not context:
                return {'success': False, 'error': 'No relevant context found'}

            # Use LLM to analyze the context
            llm_response = self._analyze_with_llm(question, context, df)

            if llm_response:
                confidence = self._calculate_semantic_confidence(
                    question, context, llm_response)
                return {
                    'success': True,
                    'result': llm_response,
                    'confidence': confidence,
                    'context': context
                }

            return {'success': False, 'error': 'LLM analysis failed'}

        except Exception as e:
            if self.debug_mode:
                print(f"Semantic analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def _try_llm_analysis(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Fall back to LLM with basic DataFrame information"""
        try:
            # Create basic context about the DataFrame
            basic_context = self._create_basic_context(df)

            # Get LLM response
            llm_response = self._analyze_with_llm(question, basic_context, df)

            if llm_response:
                return {
                    'success': True,
                    'result': llm_response,
                    'confidence': 0.6,  # Lower confidence for basic LLM analysis
                    'context': basic_context
                }

            return {'success': False, 'error': 'LLM analysis with basic context failed'}

        except Exception as e:
            if self.debug_mode:
                print(f"LLM analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_with_llm(self, question: str, context: str, df: pd.DataFrame) -> Optional[str]:
        """Use LLM to analyze context and answer question"""
        try:
            response = self.llm_client.ask(context, question)

            if self.debug_mode:
                print(f"LLM response: {response[:200]}...")

            return response

        except Exception as e:
            if self.debug_mode:
                print(f"LLM analysis error: {e}")
            return None

    def _create_basic_context(self, df: pd.DataFrame) -> str:
        """Create basic context about the DataFrame for LLM"""
        context_parts = []

        # Basic info
        context_parts.append(
            f"Dataset with {len(df)} rows and {len(df.columns)} columns")

        # Column information
        context_parts.append("Columns:")
        for col in df.columns:
            col_type = "numeric" if pd.api.types.is_numeric_dtype(
                df[col]) else "categorical"
            unique_count = df[col].nunique()
            context_parts.append(
                f"- {col} ({col_type}, {unique_count} unique values)")

        # Sample data
        context_parts.append("\nSample data:")
        sample = df.head(3)
        for _, row in sample.iterrows():
            row_desc = ", ".join(
                [f"{col}: {val}" for col, val in row.items()][:5])
            context_parts.append(f"- {row_desc}")

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            context_parts.append("\nNumeric column ranges:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                min_val, max_val = df[col].min(), df[col].max()
                context_parts.append(f"- {col}: {min_val} to {max_val}")

        return "\n".join(context_parts)

    def _calculate_structured_confidence(self, question: str, result: Dict[str, Any]) -> float:
        """Calculate confidence score for structured query results"""
        base_confidence = 0.9  # High confidence for successful structured queries

        # Reduce confidence for empty results
        if 'data' in result:
            data = result['data']
            if hasattr(data, '__len__') and len(data) == 0:
                base_confidence *= 0.7

        # Reduce confidence for very short answers (might be incomplete)
        if len(result.get('result', '')) < 20:
            base_confidence *= 0.8

        # Boost confidence for specific patterns in the question
        question_lower = question.lower()
        if any(word in question_lower for word in ['max', 'min', 'count', 'show']):
            base_confidence *= 1.1

        return min(base_confidence, 1.0)

    def _calculate_semantic_confidence(self, question: str, context: str, response: str) -> float:
        """Calculate confidence score for semantic analysis"""
        base_confidence = 0.7

        # Boost confidence if the response contains specific numbers or data
        if any(char.isdigit() for char in response):
            base_confidence += 0.1

        # Boost confidence if response mentions column names from the context
        for line in context.split('\n'):
            if 'Column' in line and ':' in line:
                col_name = line.split(':')[0].replace('Column ', '').strip()
                if col_name.lower() in response.lower():
                    base_confidence += 0.05

        # Reduce confidence for generic responses
        generic_phrases = ['i don\'t have',
                           'cannot determine', 'not enough information']
        if any(phrase in response.lower() for phrase in generic_phrases):
            base_confidence *= 0.5

        # Boost confidence for specific analysis keywords in response
        analysis_keywords = ['statistics', 'average',
                             'maximum', 'minimum', 'records']
        if any(keyword in response.lower() for keyword in analysis_keywords):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _combine_results(self, structured_result: Dict[str, Any], semantic_result: Dict[str, Any]) -> str:
        """Combine insights from structured and semantic analysis"""
        structured_answer = structured_result.get('result', '')
        semantic_answer = semantic_result.get('result', '')

        # If structured result is a simple number/value, enhance with semantic context
        if len(structured_answer) < 50 and len(semantic_answer) > 50:
            return f"{structured_answer}\n\nAdditional context: {semantic_answer}"

        # If both are substantial, present both perspectives
        elif len(structured_answer) > 50 and len(semantic_answer) > 50:
            return f"Direct analysis: {structured_answer}\n\nDetailed analysis: {semantic_answer}"

        # Default to the longer, more detailed answer
        return structured_answer if len(structured_answer) > len(semantic_answer) else semantic_answer

    def get_engine_status(self) -> Dict[str, Any]:
        """Get status information about the engines"""
        status = {
            'structured_engine': 'ready',
            'vector_engine': 'ready' if self.vector_engine.vectorizer is not None else 'not_initialized',
            'llm_client': 'ready'
        }

        if self.vector_engine.chunk_metadata:
            status['vector_chunks'] = len(self.vector_engine.chunk_metadata)
            status['chunk_types'] = self.vector_engine.get_chunk_types_summary()

        return status

    def clear_cache(self):
        """Clear all caches"""
        self.vector_engine.clear_cache()
        if self.debug_mode:
            print("Hybrid engine cache cleared")
