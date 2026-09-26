import re
from util import universal_clean
from rapidfuzz import fuzz

def normalize_address(text):
    """
    Splits address by commas, strips whitespace, and performs 
    heuristic classification of components (pincodes, roads, and regional parts).
    """
    if not text:
        return {'parts': [], 'pincodes': [], 'road': '', 'cleaned_full': ''}
    
    text_str = str(text)
    parts = [p.strip() for p in text_str.split(',') if p.strip()]
    
    # Extract postal/pincodes (5 to 6 digits) via regex
    pincodes = re.findall(r'\b\d{5,6}\b', text_str)
    
    # Heuristic separation for road/street components vs regional context
    road_keywords = {'road', 'rd', 'street', 'st', 'nagar', 'ward', 'lane', 'avenue', 'ave', 'hno', 'door', 'flat', 'plot', 'patel'}
    road_parts = []
    
    for part in parts:
        c_part = universal_clean(part)
        if any(kw in c_part.split() for kw in road_keywords):
            road_parts.append(c_part)
            
    return {
        'parts': parts,
        'pincodes': pincodes,
        'road': ' '.join(road_parts),
        'cleaned_full': universal_clean(text_str)
    }

def pin_similarity(addr1, addr2):
    """
    Returns 1.0 if a matching postal/pincode is found in both addresses, 
    otherwise returns 0.0. Acts as a strong structural anchor.
    """
    struct1 = normalize_address(addr1)
    struct2 = normalize_address(addr2)
    
    pins1 = set(struct1['pincodes'])
    pins2 = set(struct2['pincodes'])
    
    if pins1 and pins2 and (pins1 & pins2):
        return 1.0
    return 0.0

def state_similarity(addr1, addr2):
    """
    Computes regional/state component similarity using partial substring matching 
    to handle variations like abbreviations (e.g., 'UP' vs 'Uttar Pradesh').
    """
    c1 = universal_clean(addr1)
    c2 = universal_clean(addr2)
    if not c1 or not c2:
        return 0.0
    return fuzz.partial_ratio(c1, c2) / 100.0

def city_similarity(addr1, addr2):
    """
    Computes city or locality alignment using token set matching 
    to handle reordered regional components.
    """
    c1 = universal_clean(addr1)
    c2 = universal_clean(addr2)
    if not c1 or not c2:
        return 0.0
    return fuzz.token_set_ratio(c1, c2) / 100.0

def word_similarity(addr1, addr2):
    """
    Calculates overall word-level token overlap ratio between the two addresses,
    accounting for whitespace and compound word variations.
    """
    struct1 = normalize_address(addr1)
    struct2 = normalize_address(addr2)
    
    c1 = struct1['cleaned_full']
    c2 = struct2['cleaned_full']
    
    if not c1 or not c2:
        return 0.0
    
    token_set = fuzz.token_set_ratio(c1, c2) / 100.0
    sq_ratio = fuzz.ratio(c1.replace(" ", ""), c2.replace(" ", "")) / 100.0
    return float(max(token_set, sq_ratio))

def n_gram_similarity(addr1, addr2):
    """
    Calculates token-sort n-gram similarity to handle 
    reordered address chunks (e.g., Street first vs City first) and spacing differences.
    """
    struct1 = normalize_address(addr1)
    struct2 = normalize_address(addr2)
    
    c1 = struct1['cleaned_full']
    c2 = struct2['cleaned_full']
    
    if not c1 or not c2:
        return 0.0
    
    token_sort = fuzz.token_sort_ratio(c1, c2) / 100.0
    sq_ratio = fuzz.ratio(c1.replace(" ", ""), c2.replace(" ", "")) / 100.0
    return float(max(token_sort, sq_ratio))

def extract_address_feature_scores(addr1, addr2):
    """
    Extracts all address-based numerical features into a clean dictionary 
    ready to be concatenated with name features for the Neural Network.
    """
    return {
        'addr_pin_match': pin_similarity(addr1, addr2),
        'addr_state_sim': state_similarity(addr1, addr2),
        'addr_city_sim': city_similarity(addr1, addr2),
        'addr_word_sim': word_similarity(addr1, addr2),
        'addr_ngram_sim': n_gram_similarity(addr1, addr2)
    }