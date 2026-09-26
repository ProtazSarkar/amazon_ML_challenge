from collections import Counter
import math
from util import universal_clean, squeeze_text, clean_and_tokenize, compute_token_max_jaro
from rapidfuzz import fuzz

def name_tokenizer(text):
    """Cleans and tokenizes a business name into a list of words."""
    cleaned = universal_clean(text)
    return cleaned.split() if cleaned else []

def tf_idf_score(text1, text2):
    """
    Calculates TF-IDF / term frequency cosine similarity between two texts 
    across word-level tokens, character n-grams, and squeezed text n-grams.
    Handles concatenated/split words ("abc xyz" vs "abcxyz") cleanly.
    """
    c1 = universal_clean(text1)
    c2 = universal_clean(text2)
    if not c1 or not c2:
        return 0.0
    
    # 1. Word token frequency cosine similarity
    words1 = Counter(c1.split())
    words2 = Counter(c2.split())
    common_words = set(words1.keys()) & set(words2.keys())
    dot_word = sum(words1[w] * words2[w] for w in common_words)
    norm_word1 = math.sqrt(sum(v*v for v in words1.values()))
    norm_word2 = math.sqrt(sum(v*v for v in words2.values()))
    word_sim = (dot_word / (norm_word1 * norm_word2)) if (norm_word1 and norm_word2) else 0.0
    
    # 2. Squeezed text character n-gram cosine similarity (for boundary-invariant matching)
    c1_sq = c1.replace(" ", "")
    c2_sq = c2.replace(" ", "")
    if len(c1_sq) >= 2 and len(c2_sq) >= 2:
        n = 3 if min(len(c1_sq), len(c2_sq)) >= 3 else 2
        ng1 = Counter([c1_sq[i:i+n] for i in range(len(c1_sq)-n+1)])
        ng2 = Counter([c2_sq[i:i+n] for i in range(len(c2_sq)-n+1)])
        common_ng = set(ng1.keys()) & set(ng2.keys())
        dot_ng = sum(ng1[k] * ng2[k] for k in common_ng)
        norm_ng1 = math.sqrt(sum(v*v for v in ng1.values()))
        norm_ng2 = math.sqrt(sum(v*v for v in ng2.values()))
        char_sim = (dot_ng / (norm_ng1 * norm_ng2)) if (norm_ng1 and norm_ng2) else 0.0
    else:
        char_sim = 1.0 if c1_sq == c2_sq else 0.0

    return float(max(word_sim, char_sim))

def char_similarity_score(text1, text2):
    """
    Calculates character-level similarity ratio between two cleaned texts, 
    accounting for whitespace/concatenation variations.
    """
    c1 = universal_clean(text1)
    c2 = universal_clean(text2)
    if not c1 or not c2:
        return 0.0
    
    std_ratio = fuzz.ratio(c1, c2) / 100.0
    sq_ratio = fuzz.ratio(c1.replace(" ", ""), c2.replace(" ", "")) / 100.0
    return float(max(std_ratio, sq_ratio))

def word_similarity_score(text1, text2):
    """
    Calculates word-level similarity by combining token-level max Jaro-Winkler, 
    token-set ratios, and squeezed-text ratios to handle word reordering, missing suffixes,
    and concatenated words (e.g. "abc xyz" vs "abcxyz").
    """
    words1 = clean_and_tokenize(text1)
    words2 = clean_and_tokenize(text2)
    if not words1 or not words2:
        return 0.0
    
    c1 = universal_clean(text1)
    c2 = universal_clean(text2)
    c1_sq = c1.replace(" ", "")
    c2_sq = c2.replace(" ", "")
    
    # 1. Token-level max Jaro alignment from util (includes squeezed token check)
    jaro_token_score = compute_token_max_jaro(words1, words2)
    
    # 2. Token set ratio for robust word-bag matching
    token_set_score = fuzz.token_set_ratio(c1, c2) / 100.0
    
    # 3. Squeezed text ratio for word boundary invariant matching
    squeezed_ratio = fuzz.ratio(c1_sq, c2_sq) / 100.0
    
    token_avg = (jaro_token_score + token_set_score) / 2.0
    return float(max(token_avg, squeezed_ratio))

def n_gram_score(text1, text2):
    """
    Calculates n-gram similarity using token-sort ratio 
    and squeezed ratio to handle reordered name components and spacing differences.
    """
    c1 = universal_clean(text1)
    c2 = universal_clean(text2)
    if not c1 or not c2:
        return 0.0
    
    token_sort = fuzz.token_sort_ratio(c1, c2) / 100.0
    squeezed_ratio = fuzz.ratio(c1.replace(" ", ""), c2.replace(" ", "")) / 100.0
    return float(max(token_sort, squeezed_ratio))

def extract_feature_scores(text1, text2):
    """
    Extracts all name-based numerical feature scores into a dictionary 
    to feed into the neural network classifier.
    """
    c1 = universal_clean(text1)
    c2 = universal_clean(text2)
    c1_sq = c1.replace(" ", "")
    c2_sq = c2.replace(" ", "")
    
    partial_score = max(fuzz.partial_ratio(c1, c2), fuzz.partial_ratio(c1_sq, c2_sq)) / 100.0
    
    return {
        'name_tf_idf': tf_idf_score(text1, text2),
        'name_char_sim': char_similarity_score(text1, text2),
        'name_word_sim': word_similarity_score(text1, text2),
        'name_ngram_sim': n_gram_score(text1, text2),
        'name_partial': float(partial_score)
    }