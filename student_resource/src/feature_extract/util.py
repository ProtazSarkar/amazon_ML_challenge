import re
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Common legal suffixes and web/domain noise to filter out
LEGAL_SUFFIXES = {'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 'llp', 'pvt', 'private', 'llc', 'com', 'in'}

def universal_clean(text):
    """
    Handles non-Latin scripts (Devanagari, Tamil, Gujarati, etc.) via transliteration,
    strips European accents using unidecode, lowercases, converts separators to spaces 
    to preserve token boundaries, and removes punctuation.
    """
    if not text:
        return ""
    
    text_str = str(text)
    
    # 1. Handle Indian/Indic Scripts (Devanagari, Tamil, Gujarati, Bengali, etc.)
    if any('\u0900' <= char <= '\u0dff' for char in text_str):
        try:
            # Transliterate generic/ISCII script blocks to Roman ITRANS (English letters)
            text_str = transliterate(text_str, sanscript.ISCII, sanscript.ITRANS)
        except Exception:
            pass
            
    # 2. Handle European Accents & Diacritics (é -> e, ó -> o, ñ -> n)
    text_str = unidecode(text_str).lower()
    
    # 3. CRITICAL FIX: Convert separators to spaces to prevent token squashing (e.g., SMART-HALL -> smart hall)
    text_str = re.sub(r'[\-_/\.]', ' ', text_str)
    
    # 4. Strip remaining special characters and punctuation
    text_str = re.sub(r'[^a-zA-Z0-9\s]', '', text_str)
    
    # Clean up any extra multi-spaces resulting from replacements
    return ' '.join(text_str.split())

def squeeze_text(text):
    """Removes all whitespace from universally cleaned text for boundary-invariant comparison."""
    return universal_clean(text).replace(" ", "")

def clean_and_tokenize(text):
    """Cleans text using universal normalization, splits into words, and filters legal suffixes."""
    cleaned_text = universal_clean(text)
    if not cleaned_text:
        return set()
    
    words = cleaned_text.split()
    filtered_words = {w for w in words if w not in LEGAL_SUFFIXES}
    return filtered_words

def compute_token_max_jaro(words1, words2):
    """
    For every word in set 1, find the maximum Jaro-Winkler similarity 
    with any word in set 2, sum them up, and normalize.
    Also incorporates squeezed text similarity to handle split/joined word variations.
    """
    if not words1 or not words2:
        return 0.0
    
    total_score = 0.0
    for w1 in words1:
        max_sim = max(JaroWinkler.similarity(w1, w2) for w2 in words2)
        total_score += max_sim
        
    std_score = total_score / max(len(words1), len(words2))
    
    # Squeezed text comparison to correctly score concatenated/split words ("abc xyz" vs "abcxyz")
    sq1 = "".join(sorted(words1))
    sq2 = "".join(sorted(words2))
    sq_score = JaroWinkler.similarity(sq1, sq2) if (sq1 and sq2) else 0.0
    
    return float(max(std_score, sq_score))

def get_acronym(text):
    """Extracts the first letter of each word to form an acronym."""
    words = text.split()
    if len(words) > 1:
        return "".join([w[0] for w in words if w])
    return ""