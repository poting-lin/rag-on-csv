# Codebase Issues Analysis: `rag-on-csv`

## 1. Security Issues

### 1a. Regex injection in `str.contains()` calls
**Files:** `csv_qa/data_handler.py:125,369`

`str.contains()` is called with user-derived values without `regex=False`. If a user searches for a value containing regex metacharacters (e.g., `(`, `[`, `*`), it will either crash or match unintended rows.

### 1b. Temp file leak on CSV upload
**File:** `streamlit_app.py:77-79`

`tempfile.NamedTemporaryFile(delete=False)` creates temp files that are never cleaned up. Each CSV upload leaves a file on disk permanently.

### 1c. Unrestricted file path in CLI
**File:** `main.py:11`

The `-f` argument accepts any file path with no validation, allowing reading of arbitrary CSV-shaped files on the system.

---

## 2. Bugs

### 2a. Wrong attribute name breaks context filtering (Critical)
**File:** `csv_qa/question_answerer.py:349`

The code does `self.data_handler._dataframe = filtered_dataframe` but `CSVDataHandler` stores its data in `self.csv_dataframe`, not `self._dataframe`. This means the context-filtered DataFrame is never actually applied, and the original unfiltered data is always used for follow-up questions.

### 2b. Valid short LLM responses silently dropped
**File:** `csv_qa/ollama_client.py:79`

Responses shorter than 10 characters or not ending with `.`, `?`, `!` are discarded and replaced with "I don't have enough information." Valid short answers like `"42"`, `"Yes"`, or `"Lab A"` are silently dropped.

### 2c. Overly broad follow-up question detection
**File:** `csv_qa/context_memory.py:362`

The pattern `r'\b(and|but|however|instead)\b'` means virtually any compound sentence is misidentified as a follow-up question. For example, "Show records where Location is Park **and** EventType is Festival" would trigger pronoun resolution unnecessarily.

### 2d. Semantic search rebuilds index on every question
**File:** `csv_qa/hybrid_engine.py:110-116`

`_try_semantic_analysis` calls `create_structured_chunks()` and `build_vector_index()` on every question, even though the data hasn't changed. This is expensive and wasteful.

### 2e. Incorrect timedelta calculation for recency scoring
**File:** `csv_qa/context_memory.py:429`

`timedelta.seconds` only returns the seconds component (0-86399), not total seconds. For conversations spanning multiple days, the recency calculation would be incorrect. Should use `total_seconds()` instead.

---

## 3. Performance Issues

### 3a. Full DataFrame copy on every query plan
**File:** `csv_qa/data_handler.py:190`

`result_df = self.csv_dataframe.copy()` makes a full copy of the entire DataFrame for every query, even when no modifications are needed. For large CSVs, this is wasteful.

### 3b. No early termination in cross-column search
**File:** `csv_qa/data_handler.py:332-381`

`search_value_in_all_columns` iterates through every column and performs both exact and substring matching on every row, even after finding exact matches. No early termination or optimization.

### 3c. TF-IDF vectorizer rebuilt on every conversation turn
**File:** `csv_qa/context_memory.py:494-517`

`_rebuild_question_vectors()` is called after every `add_turn()`, recreating the vectorizer from scratch even for the common case of a new sequential question.

### 3d. Hard-coded 100-row limit in enhanced vector search
**File:** `csv_qa/enhanced_vector_search.py:55`

`if idx >= 100: break` means datasets with more than 100 rows have incomplete vector search coverage. This silently degrades answer quality for larger datasets with no warning to the user.

---

## 4. Code Quality / Maintainability Issues

### 4a. Duplicated conversation storage logic
**File:** `csv_qa/question_answerer.py:241-301`

The `_store_conversation_turn` pattern with `result_count`/`extracted_result_data` extraction is copy-pasted identically for all three engine types (structured, semantic, hybrid). Should be factored into a shared helper.

### 4b. Duplicated engine instances
**Files:** `csv_qa/hybrid_engine.py:16-19`, `csv_qa/question_answerer.py:42-53`

`HybridCSVEngine` creates its own `StructuredQueryEngine`, `CSVAwareVectorSearch`, and `OllamaAPIClient` internally, while `CSVQuestionAnswerer` also creates separate instances of each. Two copies of each engine exist, and the hybrid engine's copies never benefit from any state or cache from the main instances.

### 4c. Inconsistent comparison operator support
**Files:** `csv_qa/data_handler.py:134-164` vs `csv_qa/data_handler.py:204-217`

`filter_by_comparison` only handles `>`, `<`, and `=`, but `execute_query_plan` supports `>=` and `<=`. Inconsistent API surface within the same class.

### 4d. No startup validation for Ollama availability
**File:** `main.py`

The CLI creates `CSVQuestionAnswerer` without checking if Ollama is running. The user won't see an error until they ask their first question, making it unclear what went wrong during setup.

---

## 5. Robustness Issues

### 5a. Filter patterns only match single-word values
**File:** `csv_qa/structured_query_engine.py:146-148`

The regex `(\w+)\s+(?:is|equals?|=)\s+(\w+)` only captures single words. "Show records where Location is Urban Park" would only capture "Urban", missing multi-word values entirely.

### 5b. No CSV encoding handling
**File:** `csv_qa/data_handler.py:55`

`pd.read_csv(csv_path)` uses default encoding (utf-8). CSVs with different encodings (latin-1, cp1252) will crash with no helpful error message.

### 5c. Potential IndexError on empty words
**File:** `csv_qa/context_memory.py:464`

`w[0].isupper()` will throw `IndexError` if any word is an empty string (possible with multiple consecutive spaces in input).

---

## Summary

| Category | Count | Severity |
|----------|-------|----------|
| Security | 3 | Medium |
| Bugs | 5 | High (2a), Medium (others) |
| Performance | 4 | Medium |
| Code Quality | 4 | Low |
| Robustness | 3 | Medium |

**Most critical issues:**
1. **Bug 2a** — Wrong attribute name completely breaks context memory filtering
2. **Bug 2b** — Valid short LLM answers silently dropped
3. **Bug 2c** — Overly broad follow-up detection causes incorrect question routing
