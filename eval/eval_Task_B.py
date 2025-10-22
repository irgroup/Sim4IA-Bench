import os
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

def get_reciprocal_rank(sim_list, threshold=0.8):
    for i, sim in enumerate(sim_list):
        if sim >= threshold:
            return 1.0 / (i + 1)
    return 0.0  

original_file = 'simiir/evaluation_sessions_Task_B.json'
prediction_dir = 'eval/run_file_input/B'
topk = 10

with open(original_file, 'r', encoding='utf-8') as f:
    true_data = json.load(f)

true_map = {
    int(session_id): utterances[-1]
    for session_id, utterances in true_data.items()
    if session_id.isdigit() and utterances
}

model = SentenceTransformer('all-MiniLM-L6-v2')

all_results = []

for json_file in os.listdir(prediction_dir):
    if not json_file.endswith('.json'):
        continue

    file_path = os.path.join(prediction_dir, json_file)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_file} {json_file}: {e}")
        continue

    avg_sim_list = []
    avg_redundancy_list = []
    avg_jaccard_redundancy_list = []
    rds_list = []

    for qid, true_q in true_map.items():
        preds = prediction_data.get(str(qid), [])[:topk]
        while len(preds) < topk:
            preds.append("")  

        avg_terms = np.mean([len(p.split()) for p in preds]) if preds else 0.0

        valid_preds = [p for p in preds if p.strip()]  

        texts = preds + [true_q]
        embeddings = model.encode(texts, convert_to_tensor=True)
        preds_vecs = embeddings[:len(preds)]
        true_vec = embeddings[-1]
        raw_sims = util.cos_sim(preds_vecs, true_vec).cpu().numpy().flatten()
        sims = (raw_sims + 1) / 2  
        sims = list(sims) + [0.0] * (topk - len(sims))  

        threshold = 0.7   

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

        # Jaccard-based redundancy
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
        avg_sim = float(np.mean(sims))
        avg_sim_list.append(avg_sim)
        

    rds = np.mean(rds_list) * (1 - np.mean(avg_jaccard_redundancy_list))
    result = {
        "JSON File": json_file,
        "Average Similarity": np.mean(avg_sim_list),
        "Average Cosine Redundancy": np.mean(avg_redundancy_list),
        "Average Jaccard Redundancy": np.mean(avg_jaccard_redundancy_list),
        "Average Rank-Diversity Score": rds,
    }

    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df = results_df.round(3)  
results_df.to_csv("eval/results/evaluation_results_B.csv", index=False)