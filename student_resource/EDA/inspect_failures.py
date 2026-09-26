import sys
from pathlib import Path

import pandas as pd

def find_project_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / 'dataset').exists():
            return current
        current = current.parent
    return Path.cwd()

def inspect_s1_and_ground_truth():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    project_root = find_project_root()
    train_dir = project_root / 'dataset' / 'train'
    
    print("Loading datasets...")
    s1 = pd.read_csv(train_dir / 'train_source1.tsv', sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(train_dir / 'train_source2.tsv', sep='\t', dtype=str).fillna('')
    
    # Optional: Load source3 if your dataset structure includes it
    s3_path = train_dir / 'train_source3.tsv'
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str).fillna('') if s3_path.exists() else pd.DataFrame()

    # The challenge ground truth maps one S1 ID to a comma-separated ID list.
    gt_path = train_dir / 'train_ground_truth.tsv'
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        return

    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')

    # Pick a few S1 IDs that showed up as missing/zero matches in your logs
    sample_s1_ids = ['S1-102971431', 'S1-948624922', 'S1-638785454']
    s2_records = s2.set_index('entity_id').to_dict('index')
    s3_records = s3.set_index('entity_id').to_dict('index') if not s3.empty else {}
    
    print("\n" + "="*80)
    print("S1 ENTITIES VS. GROUND TRUTH MATCHES INSPECTION")
    print("="*80)
    
    for s1_id in sample_s1_ids:
        s1_row = s1[s1['entity_id'] == s1_id]
        if s1_row.empty:
            print(f"\n[S1 ID: {s1_id}] Not found in Source 1.")
            continue
            
        print(f"\nSOURCE 1 RECORD [ID: {s1_id}] | Country: {s1_row.iloc[0]['country']}")
        print(f"   Name    : {s1_row.iloc[0]['business_name']}")
        print(f"   Address : {s1_row.iloc[0]['business_address']}")
        
        # Find the single mapping row for this S1 ID.
        matched_rows = gt[gt['source1_entity_id'] == s1_id]
        
        print("\n   --- GROUND TRUTH MATCHES ---")
        if matched_rows.empty:
            print("   (No ground truth matches found for this ID in the mapping file)")
        else:
            matched_ids = [
                target_id.strip()
                for target_id in str(matched_rows.iloc[0]['matched_entity_ids']).split(',')
                if target_id.strip()
            ]
            if not matched_ids:
                print("   (Singleton: no matching S2/S3 records)")

            for target_id in matched_ids:
                target_record = s2_records.get(target_id)
                source_name = "Source 2"
                if target_record is None:
                    target_record = s3_records.get(target_id)
                    source_name = "Source 3"

                if target_record is not None:
                    print(f"   Match found in [{source_name}] -> ID: {target_id} | Country: {target_record['country']}")
                    print(f"      Name    : {target_record['business_name']}")
                    print(f"      Address : {target_record['business_address']}")
                else:
                    print(f"   WARNING: Match ID: {target_id} (Listed in Ground Truth, but record not found in S2/S3 files)")
        print("-" * 80)

if __name__ == "__main__":
    inspect_s1_and_ground_truth()