"""
Fuzzy Matching module for finding similar values in CSV data
"""
import difflib
import pandas as pd

class FuzzyMatcher:
    """
    Provides fuzzy matching capabilities to find similar values in CSV data
    """
    def __init__(self, debug_mode=False):
        """Initialize the fuzzy matcher"""
        self.debug_mode = debug_mode
        
    def calculate_similarity(self, str1, str2):
        """
        Calculate character-level similarity between two strings
        
        Combines Jaccard similarity for character sets and sequence matching ratio
        """
        # Simple character overlap ratio
        chars1 = set(str1.lower())
        chars2 = set(str2.lower())
        if not chars1 or not chars2:
            return 0
        
        # Calculate Jaccard similarity for character sets
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        # Calculate character sequence similarity
        seq_sim = difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        
        # Combine both metrics
        return (intersection / union + seq_sim) / 2
    
    def find_similar_values(self, search_term: str, df: pd.DataFrame) -> list:
        """
        Find similar values in the dataframe using multiple matching methods:
        1. Difflib sequence matching with lower cutoff (0.4)
        2. Substring matching for partial matches
        3. Word-level matching for multi-word values
        4. Character-level similarity for names and short strings
        """
        all_values = []
        original_values = {}
        
        # Collect all string values from the dataframe
        for col in df.columns:
            for val in df[col].unique():
                if isinstance(val, (str, int, float)):
                    str_val = str(val)
                    lower_val = str_val.lower()
                    all_values.append(lower_val)
                    original_values[lower_val] = str_val
        
        results = set()
        search_term_lower = search_term.lower()
        
        # Method 1: Use difflib for fuzzy matching with a lower cutoff
        similar_values = difflib.get_close_matches(search_term_lower, all_values, n=3, cutoff=0.4)
        for val in similar_values:
            results.add(original_values.get(val, val))
        
        # Method 2: Check for substring matches (if the search term is at least 3 chars)
        if len(search_term_lower) >= 3:
            for lower_val, original_val in original_values.items():
                # Check if search term is in the value or vice versa
                if (search_term_lower in lower_val) or (len(search_term_lower) > 5 and lower_val in search_term_lower):
                    results.add(original_val)
                    if len(results) >= 3:  # Limit to top 3 matches
                        break
        
        # Method 3: Check for word-level matches (for multi-word values)
        search_words = set(search_term_lower.split())
        if len(search_words) > 0:
            for lower_val, original_val in original_values.items():
                val_words = set(lower_val.split())
                # If any word matches between search and value
                if search_words.intersection(val_words) and len(search_words.intersection(val_words)) / len(search_words) > 0.5:
                    results.add(original_val)
                    if len(results) >= 3:  # Limit to top 3 matches
                        break
        
        # Method 4: Special handling for names and short strings
        if len(search_term_lower) >= 3 and len(results) < 3:
            # For each value in the dataframe
            similarity_scores = []
            for lower_val, original_val in original_values.items():
                # Calculate similarity for each word in the value
                for word in lower_val.split():
                    if len(word) >= 3:  # Only consider words of sufficient length
                        similarity = self.calculate_similarity(search_term_lower, word)
                        if similarity > 0.6:  # Higher threshold for character-level similarity
                            similarity_scores.append((similarity, original_val))
            
            # Add top matches by similarity
            for _, val in sorted(similarity_scores, reverse=True)[:3]:
                results.add(val)
                if len(results) >= 3:
                    break
        
        if self.debug_mode and results:
            print(f"Found similar values for '{search_term}': {results}")
            
        return list(results)
