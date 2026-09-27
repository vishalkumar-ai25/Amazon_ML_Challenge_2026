import numpy as np
import faiss
import pickle
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class SemanticBlockingEngine:
    def __init__(self, emb_path='embeddings/all_names_emb.npy', 
                 idx_path='embeddings/name_to_idx.pkl',
                 top_k=10, nlist=4096, nprobe=32):
        self.emb_path = emb_path
        self.idx_path = idx_path
        self.top_k = top_k
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.embeddings = None
        self.name_to_idx = None
        self.index = None
        self.candidate_ids = []
        
        self._load_data()
        
    def _load_data(self):
        logger.info(f"Loading name to index mapping from {self.idx_path}")
        with open(self.idx_path, 'rb') as f:
            self.name_to_idx = pickle.load(f)
        n_names = len(self.name_to_idx)
        logger.info(f"Loaded mapping for {n_names} names")
        
        logger.info(f"Loading embeddings from {self.emb_path}")
        self.embeddings = np.memmap(
            self.emb_path, dtype='float16', mode='r', shape=(n_names, 1536)
        )
        logger.info(f"Embeddings loaded: shape {self.embeddings.shape}, dtype {self.embeddings.dtype}")
        
    def get_embedding(self, name_lower):
        """Return embedding for a lowered name, or None if not found"""
        idx = self.name_to_idx.get(name_lower)
        if idx is not None:
            return self.embeddings[idx]
        return None
    
    def compute_cosine(self, name1_lower, name2_lower):
        """Return cosine similarity between two names. 
        Used as feature #51 during scoring.
        """
        emb1 = self.get_embedding(name1_lower)
        emb2 = self.get_embedding(name2_lower)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        emb1_f32 = emb1.astype(np.float32)
        emb2_f32 = emb2.astype(np.float32)
        return float(np.dot(emb1_f32, emb2_f32))
        
    def build_index(self, candidate_names_lower, candidate_ids):
        """Build FAISS IVFFlat index on candidate embeddings.
        candidate_names_lower: list/array of lowered business names
        candidate_ids: list/array of entity IDs or indices (e.g. 0..N-1)
        """
        logger.info(f"Building FAISS index for {len(candidate_names_lower)} candidates...")
        start_time = time.time()
        
        valid_indices = []
        self.candidate_ids = []
        
        for name, cid in zip(candidate_names_lower, candidate_ids):
            idx = self.name_to_idx.get(name)
            if idx is not None:
                valid_indices.append(idx)
                self.candidate_ids.append(cid)
        
        logger.info(f"Found embeddings for {len(valid_indices)} out of {len(candidate_names_lower)} candidates")
        
        if not valid_indices:
            logger.warning("No valid candidate embeddings found. Index will be empty.")
            return
            
        n_vectors = len(valid_indices)
        d = 1536
        
        # Chunked extraction to float32 to prevent memory spikes
        logger.info("Extracting candidate vectors in chunks...")
        cand_embs = np.empty((n_vectors, d), dtype=np.float32)
        chunk_sz = 250000
        for s in range(0, n_vectors, chunk_sz):
            e = min(s + chunk_sz, n_vectors)
            cand_embs[s:e] = self.embeddings[valid_indices[s:e]].astype(np.float32)
            
        quantizer = faiss.IndexFlatIP(d)
        actual_nlist = min(self.nlist, max(1, int(4 * np.sqrt(n_vectors)))) 
        
        logger.info(f"Creating IndexIVFFlat with nlist={actual_nlist}, metric=METRIC_INNER_PRODUCT")
        self.index = faiss.IndexIVFFlat(quantizer, d, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        
        logger.info("Training FAISS index...")
        faiss.omp_set_num_threads(32)
        if n_vectors > 1000000:
            logger.info("Sampling 500K vectors for IVF training...")
            np.random.seed(42)
            train_idx = np.random.choice(n_vectors, size=500000, replace=False)
            self.index.train(cand_embs[train_idx])
        else:
            self.index.train(cand_embs)
            
        logger.info("Adding vectors to FAISS index...")
        self.index.add(cand_embs)
        del cand_embs
        faiss.omp_set_num_threads(1)
        
        self.index.nprobe = self.nprobe
        logger.info(f"Index successfully built with {self.index.ntotal} vectors in {time.time() - start_time:.2f}s")
        
    def retrieve_semantic_candidates(self, query_names_lower, top_k=None):
        """For each query name, find top-K nearest neighbors from the index.
        Returns list of lists: [(candidate_id, cosine_score), ...]
        where candidate_id is from self.candidate_ids.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build_index first.")
            
        k = top_k if top_k is not None else self.top_k
        n_queries = len(query_names_lower)
        logger.info(f"Retrieving top {k} semantic candidates for {n_queries} queries...")
        start_time = time.time()
        
        query_indices = []
        valid_queries_mask = []
        
        for name in query_names_lower:
            idx = self.name_to_idx.get(name)
            if idx is not None:
                query_indices.append(idx)
                valid_queries_mask.append(True)
            else:
                valid_queries_mask.append(False)
                
        n_valid = len(query_indices)
        if n_valid == 0:
            logger.warning("No valid query embeddings found.")
            return [[] for _ in query_names_lower]
            
        d = 1536
        query_embs_np = np.empty((n_valid, d), dtype=np.float32)
        chunk_sz = 250000
        for s in range(0, n_valid, chunk_sz):
            e = min(s + chunk_sz, n_valid)
            query_embs_np[s:e] = self.embeddings[query_indices[s:e]].astype(np.float32)
        
        logger.info(f"Searching index for {n_valid} valid queries (batch_size=50000, 32 threads)...")
        batch_size = 50000
        all_distances = []
        all_indices = []
        
        faiss.omp_set_num_threads(32)
        for i in range(0, n_valid, batch_size):
            batch = query_embs_np[i:i+batch_size]
            distances, indices = self.index.search(batch, k)
            all_distances.append(distances)
            all_indices.append(indices)
        faiss.omp_set_num_threads(1)
            
        del query_embs_np
        distances_np = np.vstack(all_distances)
        indices_np = np.vstack(all_indices)
        
        cand_ids_arr = self.candidate_ids
        results = []
        valid_idx = 0
        
        for is_valid in valid_queries_mask:
            if is_valid:
                res = []
                for j in range(k):
                    faiss_cand_idx = indices_np[valid_idx, j]
                    if faiss_cand_idx != -1:
                        score = float(distances_np[valid_idx, j])
                        res.append((cand_ids_arr[faiss_cand_idx], score))
                results.append(res)
                valid_idx += 1
            else:
                results.append([])
                
        logger.info(f"Semantic retrieval complete for {n_queries} queries in {time.time() - start_time:.2f}s")
        return results

if __name__ == '__main__':
    print("SemanticBlockingEngine module ready.", flush=True)
