"""
Independent Semantic Matching Module for Entity Resolution
Amazon ML Challenge 2026

Provides GPU-accelerated multilingual semantic embedding scoring for hard-case
entities where lexical blocking (TF-IDF + Jaccard) fails due to cross-script
mismatches (e.g. Devanagari/Kannada vs English transliteration) or token disconnects.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params)
       intfloat/multilingual-e5-small (MIT, 118M params)
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sentence_transformers import SentenceTransformer

class SemanticEmbedder:
    """
    Batched semantic embedding computer for entity pairs on GPU.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        batch_size: int = 256
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[SemanticEmbedder] Initializing model '{model_name}' on {self.device}...")
        t0 = time.time()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.batch_size = batch_size
        print(f"[SemanticEmbedder] Model loaded in {time.time()-t0:.2f}s (fp16 on {self.device})")
        
    def encode_texts(self, texts: List[str], desc: str = "texts") -> torch.Tensor:
        """
        Encode a list of texts into L2-normalized embeddings on GPU.
        Returns: torch.Tensor of shape (N, D) on self.device.
        """
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()), device=self.device)
            
        t0 = time.time()
        # Clean empty or None strings
        clean_texts = [str(t).strip() if pd.notna(t) and str(t).strip() else "empty" for t in texts]
        
        embeddings = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device
        )
        elapsed = time.time() - t0
        throughput = len(texts) / max(elapsed, 1e-6)
        print(f"[SemanticEmbedder] Encoded {len(texts):,} {desc} in {elapsed:.2f}s ({throughput:.1f} items/sec)")
        return embeddings

def get_embedding_scores(
    s1_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = None,
    batch_size: int = 256,
    weight_name: float = 0.5,
    weight_addr: float = 0.5,
    composite: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Pluggable scoring function for entity resolution pipelines.
    
    Args:
        s1_records: DataFrame containing ['entity_id', 'business_name', 'business_address']
        candidate_records: DataFrame containing ['entity_id', 'business_name', 'business_address']
        candidate_pairs: List of (s1_id, candidate_id) pairs to score
        model_name: HuggingFace model identifier (Apache-2.0 or MIT licensed)
        device: 'cuda' or 'cpu' (defaults to cuda if available)
        batch_size: Batch size for model inference
        weight_name: Weight for business_name cosine similarity
        weight_addr: Weight for business_address cosine similarity
        composite: If True, encodes single composite string '2*name + address'
        
    Returns:
        Dict mapping s1_id -> {candidate_id: semantic_score}
    """
    t_start = time.time()
    embedder = SemanticEmbedder(model_name=model_name, device=device, batch_size=batch_size)
    
    # 1. Map IDs to record indices
    s1_map = {row['entity_id']: row for _, row in s1_records.iterrows()}
    cand_map = {row['entity_id']: row for _, row in candidate_records.iterrows()}
    
    # Collect unique entities needed
    unique_s1_ids = sorted(list({pair[0] for pair in candidate_pairs if pair[0] in s1_map}))
    unique_cand_ids = sorted(list({pair[1] for pair in candidate_pairs if pair[1] in cand_map}))
    
    s1_id_to_idx = {sid: idx for idx, sid in enumerate(unique_s1_ids)}
    cand_id_to_idx = {cid: idx for idx, cid in enumerate(unique_cand_ids)}
    
    print(f"[get_embedding_scores] Scoring {len(candidate_pairs):,} pairs across {len(unique_s1_ids):,} S1 queries and {len(unique_cand_ids):,} candidates")
    
    if composite:
        # Composite text: 2*name + " " + address
        s1_texts = [f"{s1_map[sid]['business_name']} {s1_map[sid]['business_name']} {s1_map[sid]['business_address']}" for sid in unique_s1_ids]
        cand_texts = [f"{cand_map[cid]['business_name']} {cand_map[cid]['business_name']} {cand_map[cid]['business_address']}" for cid in unique_cand_ids]
        
        s1_emb = embedder.encode_texts(s1_texts, desc="S1 composite texts")
        cand_emb = embedder.encode_texts(cand_texts, desc="Candidate composite texts")
        
        # Batch pairwise cosine computation for requested pairs
        valid_pairs = [p for p in candidate_pairs if p[0] in s1_id_to_idx and p[1] in cand_id_to_idx]
        pair_s1_indices = [s1_id_to_idx[p[0]] for p in valid_pairs]
        pair_cand_indices = [cand_id_to_idx[p[1]] for p in valid_pairs]
        
        # Batched dot product: (s1_emb[pair_s1] * cand_emb[pair_cand]).sum(dim=-1)
        sub_s1 = s1_emb[pair_s1_indices]
        sub_cand = cand_emb[pair_cand_indices]
        sim_scores = (sub_s1 * sub_cand).sum(dim=-1).clamp(0.0, 1.0).cpu().numpy()
        candidate_pairs = valid_pairs
        
    else:
        # Separate name and address embeddings
        s1_names = [s1_map[sid]['business_name'] for sid in unique_s1_ids]
        s1_addrs = [s1_map[sid]['business_address'] for sid in unique_s1_ids]
        cand_names = [cand_map[cid]['business_name'] for cid in unique_cand_ids]
        cand_addrs = [cand_map[cid]['business_address'] for cid in unique_cand_ids]
        
        s1_n_emb = embedder.encode_texts(s1_names, desc="S1 names")
        s1_a_emb = embedder.encode_texts(s1_addrs, desc="S1 addresses")
        cand_n_emb = embedder.encode_texts(cand_names, desc="Candidate names")
        cand_a_emb = embedder.encode_texts(cand_addrs, desc="Candidate addresses")
        
        valid_pairs = [p for p in candidate_pairs if p[0] in s1_id_to_idx and p[1] in cand_id_to_idx]
        pair_s1_indices = [s1_id_to_idx[p[0]] for p in valid_pairs]
        pair_cand_indices = [cand_id_to_idx[p[1]] for p in valid_pairs]
        
        sub_s1_n = s1_n_emb[pair_s1_indices]
        sub_s1_a = s1_a_emb[pair_s1_indices]
        sub_c_n = cand_n_emb[pair_cand_indices]
        sub_c_a = cand_a_emb[pair_cand_indices]
        
        sim_name = (sub_s1_n * sub_c_n).sum(dim=-1).clamp(0.0, 1.0)
        sim_addr = (sub_s1_a * sub_c_a).sum(dim=-1).clamp(0.0, 1.0)
        
        total_sim = weight_name * sim_name + weight_addr * sim_addr
        sim_scores = total_sim.cpu().numpy()
        candidate_pairs = valid_pairs
        
    # Build result dict
    scores_dict: Dict[str, Dict[str, float]] = {sid: {} for sid in unique_s1_ids}
    for (sid, cid), score in zip(candidate_pairs, sim_scores):
        scores_dict[sid][cid] = float(score)
        
    total_time = time.time() - t_start
    print(f"[get_embedding_scores] Completed in {total_time:.2f}s ({len(candidate_pairs)/max(total_time, 1e-6):.1f} pairs/sec)")
    return scores_dict
