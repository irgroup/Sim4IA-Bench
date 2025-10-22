import os
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import pyterrier as pt
import re
import sys


if not pt.started():
    pt.init()

threshold = 0.7  

def get_reciprocal_rank(sim_list, threshold=0.8):
    for i, sim in enumerate(sim_list):
        if sim >= threshold:
            return 1.0 / (i + 1)
    return 0.0  

def clean_query(q):
    q = q.replace("’", "'")         
    q = q.replace("'", "")          
    q = re.sub(r"\s+", " ", q)      
    q = re.sub(r"[^\w\s']", " ", q) 
    q = re.sub(r"\s+", " ", q).strip()
    
    return q

index_path = "./eval/index_CORE"
json_dir = './eval/run_file_input/A1'
true_file = './simiir/evaluation_sessions_Task_A.csv'
topk = 10

index = pt.IndexFactory.of(index_path)
bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=10)

model = SentenceTransformer('all-MiniLM-L6-v2')

df_true = pd.read_csv(true_file, sep=',', encoding='utf-8', usecols=['session_id', 'query'], dtype={'session_id': str})

unique_ids = pd.unique(df_true['session_id'])
id_map = {old_id: str(i+1) for i, old_id in enumerate(unique_ids)}
df_true['session_id'] = df_true['session_id'].map(id_map)

true_map = dict(zip(df_true['session_id'], df_true['query']))

all_results = []

overlap_results = []

for json_file in os.listdir(json_dir):
    print(json_file)
    if not json_file.endswith('.json'):
        continue

  
    avg_sim_list = []
    rds_list = []
    avg_overlap_list = []
    avg_redundancy_list = []
    avg_jaccard_redundancy_list = []
    file_path = os.path.join(json_dir, json_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"Processing file: {json_file}")
        data = json.load(f)

    for qid, true_q in true_map.items():
        preds = data.get(str(qid), []  )[:topk] 
        while len(preds) < topk:
            preds.append("")
             
        print(f"Original query: {true_q}")
        true_q = true_q.replace("'", "")
        query = true_q.replace('"', '')  
           
        true_df = bm25.search(clean_query(query))
        true_doc_ids = set(true_df["docno"].tolist()[:10])

        overlaps = []
        for cand in preds:
            if cand == "": 
                overlaps.append(0.0)
            else:
                cand_df = bm25.search(clean_query(cand))
                cand_doc_ids = set(cand_df["docno"].tolist()[:10])
                overlap = len(true_doc_ids.intersection(cand_doc_ids)) / 10  
                overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps)
        avg_overlap_list.append(avg_overlap)
      
        valid_preds = [p for p in preds if p.strip()]  

        texts = preds + [true_q]
        embeddings = model.encode(texts, convert_to_tensor=True)
        preds_vecs = embeddings[:len(valid_preds)]

        true_vec = embeddings[-1]
        
        sims = util.cos_sim(preds_vecs, true_vec).cpu().numpy().flatten()
        sims = (sims + 1) / 2  
        sims = list(sims) + [0.0] * (topk - len(sims))  

        rds_list.append(get_reciprocal_rank(sims, threshold=threshold))

        pairwise_sims = []
        for i in range(len(preds_vecs)):
            for j in range(i + 1, len(preds_vecs)):
                if preds[i] == "" or preds[j] == "":
                    sim = 0.0
                else:
                    sim = util.cos_sim(preds_vecs[i], preds_vecs[j])[0][0].item()
                pairwise_sims.append((sim + 1) / 2)  
        avg_redundancy = np.mean(pairwise_sims) if pairwise_sims else 0.0
        
        def jaccard(a, b):
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a and not set_b:
                return 1.0
            return len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0

        jaccard_sims = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if preds[i] == "" or preds[j] == "":
                    sim = 0.0
                else:
                    sim = jaccard(preds[i], preds[j])
                jaccard_sims.append(sim)
        avg_jaccard_redundancy = np.mean(jaccard_sims) if jaccard_sims else 0.0
        avg_jaccard_redundancy_list.append(avg_jaccard_redundancy)

        avg_redundancy_list.append(avg_redundancy)

        avg_sim = np.mean(sims)
        avg_sim_list.append(avg_sim)

        print(f"\n=== File: {json_file} | session_id = {qid} ===")
        print("True Query:")
        print(f"   {true_q}")
        print("Predictions + Similarities:")
        for i, (p, sim) in enumerate(zip(preds, sims), 1):
            print(f"{i:2d}. sim = {sim:.4f} → {p}")
        print("→ Average Similarity:  ", f"{avg_sim:.4f}")
        print("=" * 60)

    avg_sim_ = np.mean(avg_sim_list)
    avg_overlap = np.mean(avg_overlap_list)
    avg_redundancy_value = np.mean(avg_redundancy_list)
    rds = np.mean(rds_list) * (1 - np.mean(avg_jaccard_redundancy_list))
    all_results.append({
        "JSON File": json_file,
        "Average Similarity": avg_sim_,
        "Average SERP Overlap": avg_overlap,
        "Average Rank-Diversity Score": rds,
        "Average Redundancy": avg_redundancy_value,
        "Average Jaccard Redundancy": np.mean(avg_jaccard_redundancy_list),
       
    })
results_df = pd.DataFrame(all_results)

results_df = results_df.round(3)
print(results_df)

results_df.to_csv("./eval/results/evaluation_results_A1.csv", index=False)
