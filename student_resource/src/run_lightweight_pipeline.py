import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

def load_data(sample_size=1000):
    """Loads a lightweight sample of the training data for fast local execution."""
    print("Loading datasets (with sampling for speed)...")
    train_dir = os.path.join('dataset', 'train')
    
    # Load limited rows to keep it lightweight during development
    s1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size).fillna('')
    s2 = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', nrows=sample_size).fillna('')
    s3 = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', nrows=sample_size).fillna('')
    ground_truth = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t').fillna('')
    return s1, s2, s3, ground_truth

def generate_candidates(s1, s2_s3):
    """Simple blocking: matches by country and shared name tokens."""
    print("Generating candidate pairs (Blocking)...")
    candidates = []
    
    s1_dict_tokens = {}
    for _, r1 in s1.iterrows():
        id1 = r1['entity_id']
        tokens1 = set(str(r1['business_name']).lower().split())
        s1_dict_tokens[id1] = {
            'tokens': tokens1,
            'country': str(r1['country']).strip()
        }
        
    for _, r2 in s2_s3.iterrows():
        id2 = r2['entity_id']
        country2 = str(r2['country']).strip()
        tokens2 = set(str(r2['business_name']).lower().split())
        
        for id1, data1 in s1_dict_tokens.items():
            if data1['country'] == country2 and data1['country'] != '':
                if data1['tokens'].intersection(tokens2):
                    candidates.append({
                        'source1_entity_id': id1,
                        'candidate_entity_id': id2
                    })
                    
    return pd.DataFrame(candidates)

def extract_features(s1, s2_s3, candidates):
    """Computes TF-IDF character similarity features."""
    print("Extracting features...")
    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2_s3_dict = s2_s3.set_index('entity_id').to_dict('index')
    
    all_names = pd.concat([s1['business_name'], s2_s3['business_name']]).astype(str).tolist()
    name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    name_vectorizer.fit(all_names)
    
    all_addrs = pd.concat([s1['business_address'], s2_s3['business_address']]).astype(str).tolist()
    addr_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    addr_vectorizer.fit(all_addrs)
    
    feature_rows = []
    for _, row in candidates.iterrows():
        id1 = row['source1_entity_id']
        id2 = row['candidate_entity_id']
        
        rec1 = s1_dict[id1]
        rec2 = s2_s3_dict[id2]
        
        name1, name2 = str(rec1['business_name']), str(rec2['business_name'])
        addr1, addr2 = str(rec1['business_address']), str(rec2['business_address'])
        
        name_sim = float(cosine_similarity(name_vectorizer.transform([name1]), name_vectorizer.transform([name2]))[0, 0])
        
        if addr1.strip() and addr2.strip():
            addr_sim = float(cosine_similarity(addr_vectorizer.transform([addr1]), addr_vectorizer.transform([addr2]))[0, 0])
        else:
            addr_sim = 0.0
            
        feature_rows.append({
            'source1_entity_id': id1,
            'candidate_entity_id': id2,
            'name_similarity': name_sim,
            'address_similarity': addr_sim
        })
        
    return pd.DataFrame(feature_rows)

if __name__ == "__main__":
    # Load a lightweight sample (e.g., 500 records) to test quickly
    s1, s2, s3, gt = load_data(sample_size=500)
    s2_s3 = pd.concat([s2, s3], ignore_index=True)
    
    # 1. Generate Candidates
    candidates_df = generate_candidates(s1, s2_s3)
    os.makedirs('output', exist_ok=True)
    
    if len(candidates_df) > 0:
        # Save candidate_pairs.tsv
        cand_grouped = candidates_df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
        cand_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
        cand_grouped.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
        print("Saved output/candidate_pairs.tsv successfully.")
        
        # 2. Extract Features & Label with Ground Truth
        features_df = extract_features(s1, s2_s3, candidates_df)
        
        # Build training labels from ground truth
        gt_mapping = {}
        for _, row in gt.iterrows():
            matches = str(row['matched_entity_ids']).split(',')
            for m in matches:
                if m.strip():
                    gt_mapping[(row['source1_entity_id'], m.strip())] = 1
                    
        features_df['label'] = features_df.apply(lambda r: gt_mapping.get((r['source1_entity_id'], r['candidate_entity_id']), 0), axis=1)
        
        # 3. Train a Lightweight Classifier
        X = features_df[['name_similarity', 'address_similarity']]
        y = features_df['label']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        if y.nunique() < 2:
            # A small sample can contain no true candidate matches, so there is
            # no positive-class probability column to select from.
            only_label = int(y.iloc[0])
            print(
                f"Warning: training labels contain only class {only_label}; "
                "using a constant match probability."
            )
            features_df['match_prob'] = float(only_label)
        else:
            model.fit(X, y)
            positive_class = list(model.classes_).index(1)
            features_df['match_prob'] = model.predict_proba(X)[:, positive_class]
        
        # Threshold Tuning for Precision-heavy F_0.5 (e.g., threshold > 0.70)
        threshold = 0.70
        matched_results = []
        
        for s1_id, group in features_df.groupby('source1_entity_id'):
            valid_matches = group[group['match_prob'] >= threshold]['candidate_entity_id'].tolist()
            matched_results.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': ','.join(valid_matches) if valid_matches else ''
            })
            
        results_df = pd.DataFrame(matched_results)
        
        # Ensure all S1 entities appear even if missing from candidates (singletons)[cite: 1]
        all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id']})
        final_results = pd.merge(all_s1, results_df, on='source1_entity_id', how='left').fillna('')
        
        # Save matching_results.tsv
        final_results.to_csv('output/matching_results.tsv', sep='\t', index=False)
        os.makedirs('model', exist_ok=True)
        with open(os.path.join('model', 'randomforest_model.pkl'), 'wb') as model_file:
            pickle.dump(
                {
                    'model': model,
                    'feature_columns': ['name_similarity', 'address_similarity'],
                    'threshold': threshold,
                    'training_label_counts': y.value_counts().to_dict(),
                },
                model_file,
            )
        print("Saved model/randomforest_model.pkl.")
        print("Saved output/matching_results.tsv successfully[cite: 1].")
    else:
        print("No candidates found with current small sample settings. Try increasing sample_size.")