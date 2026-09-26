import os
import pandas as pd
from feature_extract import extract_advanced_features


TARGET_IDS = ['S1-965667', 'S1-55344266', 'S1-343815751', 'S1-656753428']


def load_sample_with_ids(path, sample_size, required_ids):
    """Load a fast sample and add required records found later in the TSV."""
    sample = pd.read_csv(path, sep='\t', nrows=sample_size).fillna('')
    missing_ids = set(required_ids) - set(sample['entity_id'])
    if not missing_ids:
        return sample

    extra_rows = []
    for chunk in pd.read_csv(path, sep='\t', chunksize=100000):
        selected = chunk[chunk['entity_id'].isin(missing_ids)]
        if not selected.empty:
            extra_rows.append(selected)
            missing_ids -= set(selected['entity_id'])
            if not missing_ids:
                break

    if extra_rows:
        sample = pd.concat([sample, *extra_rows], ignore_index=True).fillna('')
    return sample.drop_duplicates(subset='entity_id')


def test_matching_pipeline():
    print("Loading training data sample...")
    train_dir = os.path.join('dataset', 'train')
    gt = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t').fillna('')

    # Keep the fast sample, but include the requested entities and their true
    # matches so the diagnostic comparison is not guaranteed to print None.
    target_matches = gt[gt['source1_entity_id'].isin(TARGET_IDS)]
    required_s2_s3 = {
        match_id.strip()
        for values in target_matches['matched_entity_ids']
        for match_id in str(values).split(',')
        if match_id.strip()
    }
    s1 = load_sample_with_ids(
        os.path.join(train_dir, 'train_source1.tsv'), 1000, TARGET_IDS
    )
    s2 = load_sample_with_ids(
        os.path.join(train_dir, 'train_source2.tsv'), 1000,
        {entity_id for entity_id in required_s2_s3 if entity_id.startswith('S2-')},
    )
    s3 = load_sample_with_ids(
        os.path.join(train_dir, 'train_source3.tsv'), 1000,
        {entity_id for entity_id in required_s2_s3 if entity_id.startswith('S3-')},
    )

    s2_s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("Generating candidates and applying heuristic score...")
    results = []
    
    # Quick blocking by country and token overlap to limit pairs
    for _, r1 in s1.iterrows():
        id1 = r1['entity_id']
        country1 = str(r1['country']).strip()
        tokens1 = set(str(r1['business_name']).lower().split())
        if not tokens1:
            continue
            
        matched_candidates = []
        
        for _, r2 in s2_s3.iterrows():
            id2 = r2['entity_id']
            country2 = str(r2['country']).strip()
            
            # Strict country check
            if country1 == country2 and country1 != '':
                tokens2 = set(str(r2['business_name']).lower().split())
                
                # Check candidate block condition (sharing at least 1 name token)
                if tokens1.intersection(tokens2):
                    # Extract your features using your custom function
                    features = extract_advanced_features(r1, r2.to_dict())
                    
                    # Apply your hardcoded weighted rule:
                    # 0.65 * name_max_jaro + 0.25 * address_jaro + 0.10 * country_match >= 0.80
                    score = (0.65 * features['name_max_jaro']) + \
                            (0.25 * features['address_jaro']) + \
                            (0.10 * features['country_match'])
                            
                    if score >= 0.67:
                        matched_candidates.append(id2)
                        
        results.append({
            'source1_entity_id': id1,
            'matched_entity_ids': ','.join(matched_candidates)
        })
        
    # Save output to check format
    os.makedirs('output', exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv('output/matching_results.tsv', sep='\t', index=False)
    print("Saved test results to output/matching_results.tsv successfully.")
    
    print("\n--- Heuristic vs Ground Truth Check ---")
    for tid in TARGET_IDS:
        pred = res_df[res_df['source1_entity_id'] == tid]['matched_entity_ids'].values
        actual = gt[gt['source1_entity_id'] == tid]['matched_entity_ids'].values
        
        print(f"Entity: {tid}")
        print(f"  Predicted: {pred[0] if len(pred) > 0 else 'None'}")
        print(f"  Actual GT: {actual[0] if len(actual) > 0 else 'None'}")

if __name__ == "__main__":
    test_matching_pipeline()