#!/usr/bin/env python3
"""
Testing & Inference Script for Entity Resolution Pipeline
=========================================================
Loads the pre-trained model state dict from model/nn_model.pkl, runs inference on the
first 500 samples of the TEST dataset (`dataset/test/`), compares predictions 
against ground truth (if available), and generates:
  - student_resource/output/test_match.tsv
  - student_resource/output/test_match.tst
  - student_resource/output/candidate_pairs.tsv
  - student_resource/output/test_candidate_pairs.tsv
"""

import os
import sys
import pickle
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import torch

# Ensure src/ and feature_extract/ are in sys.path
def find_project_root():
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    for root in [cwd, script_dir, script_dir.parent, script_dir.parent.parent]:
        if (root / 'student_resource' / 'src').exists():
            return (root / 'student_resource').resolve()
        if (root / 'src').exists():
            return root.resolve()
    return script_dir.parent.resolve()

sr_root = find_project_root()
src_dir = sr_root / 'src'
feature_dir = sr_root / 'src' / 'feature_extract'

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(feature_dir) not in sys.path:
    sys.path.insert(0, str(feature_dir))

# Import feature extraction modules
import addr_feature_extract
import name_feature_extract
import util
from nn_model import BranchedEntityMatcher

extract_address_feature_scores = addr_feature_extract.extract_address_feature_scores
extract_feature_scores = name_feature_extract.extract_feature_scores
universal_clean = util.universal_clean
clean_and_tokenize = util.clean_and_tokenize

NAME_FEATURES = [
    'name_tf_idf', 'name_char_sim', 'name_word_sim', 'name_ngram_sim', 'name_partial'
]
ADDRESS_FEATURES = [
    'addr_pin_match', 'addr_state_sim', 'addr_city_sim', 'addr_word_sim', 'addr_ngram_sim'
]

def load_pretrained_model():
    """Loads the pre-trained neural network model from model/nn_model.pkl safely."""
    print("=" * 80)
    print("STEP 1: Loading Pre-trained Model from model/nn_model.pkl...")
    print("=" * 80)
    
    model_path = sr_root.parent / 'model' / 'nn_model.pkl' if not (sr_root / 'model' / 'nn_model.pkl').exists() else sr_root / 'model' / 'nn_model.pkl'
    if not model_path.exists():
        model_path = sr_root / 'model' / 'nn_model.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure model/nn_model.pkl exists.")

    print(f"Found model at: {model_path}")
    
    model = BranchedEntityMatcher(name_features=len(NAME_FEATURES), address_features=len(ADDRESS_FEATURES))
    
    try:
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
    except Exception:
        loaded_obj = torch.load(model_path, map_location=torch.device('cpu'))

    if isinstance(loaded_obj, dict):
        if 'model' in loaded_obj:
            checkpoint_model = loaded_obj['model']
            if isinstance(checkpoint_model, torch.nn.Module):
                model = checkpoint_model
            elif isinstance(checkpoint_model, dict):
                model.load_state_dict(checkpoint_model)
            else:
                raise TypeError(
                    f"Checkpoint 'model' entry has unsupported type {type(checkpoint_model)}."
                )
        elif 'state_dict' in loaded_obj:
            model.load_state_dict(loaded_obj['state_dict'])
        elif 'model_state_dict' in loaded_obj:
            model.load_state_dict(loaded_obj['model_state_dict'])
        else:
            model.load_state_dict(loaded_obj)
    elif hasattr(loaded_obj, 'eval'):
        model = loaded_obj
    else:
        raise TypeError(f"Loaded object from {model_path} is of type {type(loaded_obj)} and could not be loaded into the model.")

    model.eval()
    print("Model loaded successfully.")
    return model

def run_test_inference(model, test_dir: Path, sample_size: int = 500):
    """Runs candidate generation, feature extraction, and prediction on the TEST dataset."""
    print("\n" + "=" * 80)
    print("STEP 2: Running Inference on TEST Dataset (First 500 Samples)...")
    print("=" * 80)
    print(f"Loading TEST dataset from: {test_dir}")

    s1_path = test_dir / 'test_source1.tsv'
    s2_path = test_dir / 'test_source2.tsv'
    s3_path = test_dir / 'test_source3.tsv'

    s1_test = pd.read_csv(s1_path, sep='\t', nrows=sample_size).fillna('')
    s2_test = pd.read_csv(s2_path, sep='\t', nrows=sample_size * 10).fillna('')
    s3_test = pd.read_csv(s3_path, sep='\t', nrows=sample_size * 10).fillna('')

    print(f"Loaded TEST S1 ({len(s1_test)} entities), S2 ({len(s2_test)} entities), S3 ({len(s3_test)} entities).")

    s2_s3_test = pd.concat([s2_test, s3_test], ignore_index=True).drop_duplicates(subset='entity_id')
    s2_s3_dict = s2_s3_test.set_index('entity_id').to_dict('index')

    print("Generating candidate pairs for TEST set...")
    token_index = defaultdict(list)
    sq_index = defaultdict(list)

    for id2, r2 in s2_s3_dict.items():
        bname2 = str(r2.get('business_name', ''))
        cname2 = universal_clean(bname2)
        tokens2 = clean_and_tokenize(bname2)
        sq2 = cname2.replace(" ", "")

        for t in tokens2:
            token_index[t].append(id2)
        if sq2:
            sq_index[sq2].append(id2)

    test_candidates = []
    seen_pairs = set()

    for _, r1 in s1_test.iterrows():
        id1 = r1['entity_id']
        country1 = str(r1.get('country', '')).strip()
        bname1 = str(r1.get('business_name', ''))
        baddr1 = str(r1.get('business_address', ''))

        cname1 = universal_clean(bname1)
        tokens1 = clean_and_tokenize(bname1)
        sq1 = cname1.replace(" ", "")

        candidate_ids = set()
        for t in tokens1:
            matches = token_index.get(t, [])
            if len(matches) <= 80:
                candidate_ids.update(matches)
        if sq1:
            candidate_ids.update(sq_index.get(sq1, []))

        for id2 in candidate_ids:
            if (id1, id2) in seen_pairs:
                continue
            seen_pairs.add((id1, id2))

            r2 = s2_s3_dict[id2]
            country2 = str(r2.get('country', '')).strip()
            if country1 != country2 or not country1:
                continue

            name_feats = extract_feature_scores(bname1, str(r2.get('business_name', '')))
            addr_feats = extract_address_feature_scores(baddr1, str(r2.get('business_address', '')))

            test_candidates.append({
                'source1_entity_id': id1,
                'candidate_entity_id': id2,
                **name_feats,
                **addr_feats,
                'country_match': 1
            })

    df_test_candidates = pd.DataFrame(test_candidates)
    print(f"Generated {len(df_test_candidates)} candidate pairs for TEST dataset.")

    # Format Candidate Pairs TSV
    cand_pairs_grouped = []
    if not df_test_candidates.empty:
        for s1_id, group in df_test_candidates.groupby('source1_entity_id'):
            c_list = group['candidate_entity_id'].tolist()
            cand_pairs_grouped.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ','.join(c_list)
            })
    df_cand_pairs = pd.DataFrame(cand_pairs_grouped)

    # Model Inference
    if not df_test_candidates.empty:
        model.eval()
        name_tensor = torch.tensor(df_test_candidates[NAME_FEATURES].to_numpy(dtype=np.float32))
        address_tensor = torch.tensor(df_test_candidates[ADDRESS_FEATURES].to_numpy(dtype=np.float32))
        country_tensor = torch.tensor(df_test_candidates[['country_match']].to_numpy(dtype=np.float32))

        with torch.no_grad():
            logits = model(name_tensor, address_tensor, country_tensor)
            df_test_candidates['match_prob'] = torch.sigmoid(logits).numpy()

        threshold = 0.50
        matching_results = []
        for s1_id, group in df_test_candidates.groupby('source1_entity_id'):
            matched = group[group['match_prob'] >= threshold]['candidate_entity_id'].tolist()
            matching_results.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': ','.join(matched) if matched else ''
            })
        df_matches = pd.DataFrame(matching_results)
    else:
        df_matches = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])

    # Ensure all S1 test entities exist in final outputs (singletons included)
    all_test_s1 = pd.DataFrame({'source1_entity_id': s1_test['entity_id']})
    final_matches = pd.merge(all_test_s1, df_matches, on='source1_entity_id', how='left').fillna('')
    final_candidates = pd.merge(all_test_s1, df_cand_pairs, on='source1_entity_id', how='left').fillna('')

    # Save to output folder
    out_dir = sr_root / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    test_match_tsv = out_dir / 'test_match.tsv'
    test_match_tst = out_dir / 'test_match.tst'
    candidate_pairs_tsv = out_dir / 'candidate_pairs.tsv'
    test_candidate_pairs_tsv = out_dir / 'test_candidate_pairs.tsv'

    final_matches.to_csv(test_match_tsv, sep='\t', index=False)
    final_matches.to_csv(test_match_tst, sep='\t', index=False)
    final_candidates.to_csv(candidate_pairs_tsv, sep='\t', index=False)
    final_candidates.to_csv(test_candidate_pairs_tsv, sep='\t', index=False)

    print("\n" + "=" * 80)
    print("STEP 3: Output Saved Successfully!")
    print("=" * 80)
    print(f" Saved Matching Results : {test_match_tsv}")
    print(f" Saved Matching (.tst)  : {test_match_tst}")
    print(f" Saved Candidates       : {candidate_pairs_tsv}")
    print(f" Saved Test Candidates  : {test_candidate_pairs_tsv}")
    print(f" Total Evaluated Test S1: {len(final_matches)}")
    print(f" S1 Entities with Match : {(final_matches['matched_entity_ids'] != '').sum()}")

    # Compare Predicted vs Actual Ground Truth if available in test set
    gt_path = test_dir / 'test_ground_truth.tsv'
    if gt_path.exists():
        print("\n" + "-" * 40)
        print("Comparing Predicted Output with Actual Ground Truth...")
        gt_df = pd.read_csv(gt_path, sep='\t').fillna('')
        gt_map = {}
        for s1_id, m_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
            matches = set(m.strip() for m in str(m_str).split(',') if m.strip())
            gt_map[s1_id] = matches

        true_pos = 0
        pred_pos = 0
        actual_pos = 0

        for _, row in final_matches.iterrows():
            s1_id = row['source1_entity_id']
            preds = set(m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip())
            actuals = gt_map.get(s1_id, set())

            true_pos += len(preds.intersection(actuals))
            pred_pos += len(preds)
            actual_pos += len(actuals)

        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        recall = true_pos / actual_pos if actual_pos > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f" Evaluation Metrics:")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall:    {recall:.4f}")
        print(f"   - F1-Score:  {f1:.4f}")
        print("-" * 40)
    print("=" * 80)

def main():
    t0 = time.time()
    test_dir = sr_root / 'dataset' / 'test'

    model = load_pretrained_model()
    run_test_inference(model, test_dir, sample_size=500)
    print(f"\nExecution completed in {time.time() - t0:.2f} seconds.")

if __name__ == '__main__':
    main()