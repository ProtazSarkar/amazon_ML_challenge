import re
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Common legal suffixes to strip out (both English and common variations)
LEGAL_SUFFIXES = {'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 'llp', 'pvt', 'private', 'llc'}

def universal_clean(text):
    """
    Handles non-Latin scripts (Devanagari, Tamil, Gujarati, etc.) via transliteration,
    strips European accents using unidecode, lowercases, and converts punctuation/separators to spaces.
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
    
    # 3. Convert separators to spaces to preserve token boundaries (e.g. SMART-HALL -> smart hall)
    text_str = re.sub(r'[\-_/\.]', ' ', text_str)
    
    # 4. Strip special characters and punctuation
    text_str = re.sub(r'[^a-zA-Z0-9\s]', '', text_str)
    
    return ' '.join(text_str.split())

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
    Incorporate squeezed text similarity to handle split/joined word variations.
    """
    if not words1 or not words2:
        return 0.0
    
    total_score = 0.0
    for w1 in words1:
        max_sim = max(JaroWinkler.similarity(w1, w2) for w2 in words2)
        total_score += max_sim
        
    std_score = total_score / max(len(words1), len(words2))
    
    sq1 = "".join(sorted(words1))
    sq2 = "".join(sorted(words2))
    sq_score = JaroWinkler.similarity(sq1, sq2) if (sq1 and sq2) else 0.0
    
    return float(max(std_score, sq_score))

def extract_advanced_features(rec1, rec2):
    """Extracts core robust similarity features including cross-lingual and cross-script support."""
    name1, name2 = str(rec1.get('business_name', '')), str(rec2.get('business_name', ''))
    addr1, addr2 = str(rec1.get('business_address', '')), str(rec2.get('business_address', ''))
    
    # 1. Clean, transliterate, and tokenize names
    set1 = clean_and_tokenize(name1)
    set2 = clean_and_tokenize(name2)
    name_max_jaro = compute_token_max_jaro(set1, set2)
    
    # 2. Address Jaro Similarity (with universal cleaning applied)
    clean_addr1 = universal_clean(addr1)
    clean_addr2 = universal_clean(addr2)
    
    if clean_addr1.strip() and clean_addr2.strip():
        address_jaro = float(max(
            JaroWinkler.similarity(clean_addr1, clean_addr2),
            JaroWinkler.similarity(clean_addr1.replace(" ", ""), clean_addr2.replace(" ", ""))
        ))
    else:
        address_jaro = 0.0
        
    # 3. Country Match Indicator
    country_match = int(str(rec1.get('country', '')).strip() == str(rec2.get('country', '')).strip())
    
    return {
        'name_max_jaro': name_max_jaro,
        'address_jaro': address_jaro,
        'country_match': country_match
    }