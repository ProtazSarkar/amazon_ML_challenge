import importlib
import numpy as np
import os
import pickle
import sys

import pandas as pd
import torch
from nn_model import BranchedEntityMatcher


FEATURE_DIR = os.path.join(os.path.dirname(__file__), 'feature_extract')
if FEATURE_DIR not in sys.path:
    sys.path.insert(0, FEATURE_DIR)

address_feature_module = importlib.import_module('addr_feature_extract')
name_feature_module = importlib.import_module('name_feature_extract')
util_module = importlib.import_module('util')

extract_address_feature_scores = address_feature_module.extract_address_feature_scores
extract_feature_scores = name_feature_module.extract_feature_scores
universal_clean = util_module.universal_clean
clean_and_tokenize = util_module.clean_and_tokenize



NAME_FEATURES = [
    'name_tf_idf', 'name_char_sim', 'name_word_sim', 'name_ngram_sim', 'name_partial'
]
ADDRESS_FEATURES = [
    'addr_pin_match', 'addr_state_sim', 'addr_city_sim', 'addr_word_sim', 'addr_ngram_sim'
]


def load_sample_with_ids(path, sample_size, required_ids):
    """Load the fast prefix sample and add required records from later rows."""
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    sample = pd.read_csv(path, sep='\t', nrows=sample_size, usecols=lambda c: c in cols).fillna('')
    missing_ids = set(required_ids) - set(sample['entity_id'])
    if not missing_ids:
        return sample

    extra_rows = []
    for chunk in pd.read_csv(path, sep='\t', chunksize=150000, usecols=lambda c: c in cols):
        selected = chunk[chunk['entity_id'].isin(missing_ids)]
        if not selected.empty:
            extra_rows.append(selected)
            missing_ids -= set(selected['entity_id'])
            if not missing_ids:
                break

    if extra_rows:
        sample = pd.concat([sample, *extra_rows], ignore_index=True).fillna('')
    return sample.drop_duplicates(subset='entity_id')



def run_neural_network_pipeline():
    print("Loading training data...")
    train_dir = os.path.join('dataset', 'train')
    gt = pd.read_csv(
        os.path.join(train_dir, 'train_ground_truth.tsv'), 
        sep='\t', 
        usecols=['source1_entity_id', 'matched_entity_ids']
    ).fillna('')

    sample_size = 500
    s1 = pd.read_csv(
        os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size
    ).fillna('')

    # Build ground truth mapping and identify required target IDs for sampled s1 entities
    print("Building ground truth mapping...", flush=True)
    from collections import defaultdict
    sampled_s1_ids = set(s1['entity_id'])
    required_target_ids = set()
    gt_mapping = {}
    gt_by_s1 = defaultdict(list)
    for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        matches = str(m_str).split(',')
        for m in matches:
            m_clean = m.strip()
            if m_clean:
                gt_mapping[(s1_id, m_clean)] = 1
                gt_by_s1[s1_id].append(m_clean)
                if s1_id in sampled_s1_ids:
                    required_target_ids.add(m_clean)

    print(f"Loading candidate target records (Sample size {sample_size} + Ground Truth targets)...", flush=True)
    s2 = load_sample_with_ids(
        os.path.join(train_dir, 'train_source2.tsv'), sample_size, required_target_ids
    )
    s3 = load_sample_with_ids(
        os.path.join(train_dir, 'train_source3.tsv'), sample_size, required_target_ids
    )

    s2_s3 = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset='entity_id')
    
    print("Generating candidate pairs and extracting multi-signal features...", flush=True)
    token_index = defaultdict(list)
    sq_index = defaultdict(list)
    s2_s3_dict = {}

    for id2, c2, bname2, baddr2 in zip(s2_s3['entity_id'], s2_s3['country'], s2_s3['business_name'], s2_s3['business_address']):
        c2_str = str(c2).strip()
        cname2 = universal_clean(bname2)
        tokens2 = clean_and_tokenize(bname2)
        sq2 = cname2.replace(" ", "")
        
        rec2 = {
            'entity_id': id2,
            'country': c2_str,
            'business_name': bname2,
            'business_address': baddr2,
            'tokens': tokens2,
            'sq': sq2
        }
        s2_s3_dict[id2] = rec2
        
        for t in tokens2:
            token_index[t].append(id2)
        if sq2:
            sq_index[sq2].append(id2)

    training_samples = []
    seen_pairs = set()

    for id1, country1_raw, bname1, baddr1 in zip(s1['entity_id'], s1['country'], s1['business_name'], s1['business_address']):
        country1 = str(country1_raw).strip()
        cname1 = universal_clean(bname1)
        tokens1 = clean_and_tokenize(bname1)
        sq1 = cname1.replace(" ", "")

        if not tokens1 and not sq1:
            continue
            
        candidate_ids = set()
        
        # 1. Exact token overlap candidates (skipping overly generic stopwords with > 80 matches)
        for t in tokens1:
            matches = token_index.get(t, [])
            if len(matches) <= 80:
                candidate_ids.update(matches)

        # 2. Squeezed text exact match candidates ("abc xyz" vs "abcxyz")
        if sq1:
            candidate_ids.update(sq_index.get(sq1, []))
            
        # 3. Ground Truth true matches for id1 (fast dict lookup)
        for m_target in gt_by_s1.get(id1, []):
            if m_target in s2_s3_dict:
                candidate_ids.add(m_target)

        # Extract features for candidate pairs
        for id2 in candidate_ids:
            if (id1, id2) in seen_pairs:
                continue
            seen_pairs.add((id1, id2))
            
            r2 = s2_s3_dict[id2]
            country2 = r2['country']
            
            is_true_match = (id1, id2) in gt_mapping
            country_ok = (country1 == country2 and country1 != '') or is_true_match
            if not country_ok:
                continue

            name_features = extract_feature_scores(bname1, r2['business_name'])
            address_features = extract_address_feature_scores(baddr1, r2['business_address'])
            
            label = gt_mapping.get((id1, id2), 0)
            
            training_sample = {
                'source1_entity_id': id1,
                'candidate_entity_id': id2,
                **name_features,
                **address_features,
                'country_match': int(country1 == country2),
                'label': label
            }
            training_samples.append(training_sample)

    df_features = pd.DataFrame(training_samples)
    print(f"Total candidate pairs generated: {len(df_features)}", flush=True)


    
    # Train separate name and address branches, then fuse both with country.
    print("Training branched Neural Network model on extracted features...")
    name_tensor = torch.tensor(df_features[NAME_FEATURES].to_numpy(dtype=np.float32))
    address_tensor = torch.tensor(df_features[ADDRESS_FEATURES].to_numpy(dtype=np.float32))
    country_tensor = torch.tensor(
        df_features[['country_match']].to_numpy(dtype=np.float32)
    )
    y = df_features['label']
    print(f"Training label counts: {y.value_counts().to_dict()}")

    torch.manual_seed(42)
    nn_model = BranchedEntityMatcher(
        name_features=len(NAME_FEATURES), address_features=len(ADDRESS_FEATURES)
    )
    labels_tensor = torch.tensor(y.to_numpy(dtype=np.float32))
    positive_count = float(labels_tensor.sum())
    negative_count = float(len(labels_tensor) - positive_count)
    positive_weight = negative_count / positive_count if positive_count else 1.0
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight)
    )
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.005)

    nn_model.train()
    for epoch in range(60):
        optimizer.zero_grad()
        logits = nn_model(name_tensor, address_tensor, country_tensor)
        loss = loss_function(logits, labels_tensor)
        loss.backward()
        optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        logits = nn_model(name_tensor, address_tensor, country_tensor)
        df_features['match_prob'] = torch.sigmoid(logits).numpy()
    
    # Decision Threshold
    threshold = 0.50
    results = []
    
    for s1_id, group in df_features.groupby('source1_entity_id'):
        valid_matches = group[group['match_prob'] >= threshold]['candidate_entity_id'].tolist()
        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(valid_matches) if valid_matches else ''
        })
        
    # Ensure all Source 1 entities are present (handling singletons)
    res_df = pd.DataFrame(results)
    all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    final_results = pd.merge(all_s1, res_df, on='source1_entity_id', how='left').fillna('')
    
    os.makedirs('output', exist_ok=True)
    final_results.to_csv('output/matching_results.tsv', sep='\t', index=False)
    os.makedirs('model', exist_ok=True)
    with open(os.path.join('model', 'nn_model.pkl'), 'wb') as model_file:
        pickle.dump(
            {
                'model': nn_model,
                'name_feature_columns': NAME_FEATURES,
                'address_feature_columns': ADDRESS_FEATURES,
                'feature_columns': NAME_FEATURES + ADDRESS_FEATURES + ['country_match'],
                'architecture': 'name_branch + address_branch + country_similarity -> fusion',
                'threshold': threshold,
                'training_label_counts': y.value_counts().to_dict(),
            },
            model_file,
        )
    print("Saved model/nn_model.pkl.")
    print("Neural Network training complete! Saved predictions to output/matching_results.tsv.")

if __name__ == "__main__":
    run_neural_network_pipeline()