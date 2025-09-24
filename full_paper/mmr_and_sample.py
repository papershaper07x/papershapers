"""
mmr_and_sample.py

Small, well-documented utilities for MMR re-ranking and stratified sampling
from a candidate pool. Useful for selecting diverse, relevant snippets from
Pinecone retrieval results before sending them to the generator.

Functions:
 - mmr_rerank(query_emb, candidate_embs, candidates, top_k, diversity, return_scores=False)
 - stratified_sample_by_metadata(query_emb, candidate_embs, candidates, metadata, chapter_weights, n_samples, diversity, seed)
 - mmr_and_stratified_sample(...)  # convenience wrapper

Usage:
  - Import the module and call `mmr_and_stratified_sample(...)` with numpy
    arrays for embeddings and a list of metadata dicts for candidates.

Note: This module assumes embeddings are L2-normalized (unit vectors). If
not normalized, cosine similarities will still work but normalization is
recommended for stable behaviour.

"""

from typing import List, Dict, Tuple, Optional
import numpy as np

# ----------------------
# Utilities
# ----------------------


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix between rows of `a` and rows of `b`.

    Assumes a and b are 2D arrays of shape (n, d) and (m, d).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D arrays: (n,d) and (m,d)")
    # If embeddings are normalized, dot product == cosine similarity
    return np.dot(a, b.T)


# ----------------------
# MMR implementation
# ----------------------


def mmr_rerank(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: List[dict],
    top_k: int,
    diversity: float = 0.7,
    seed: Optional[int] = None,
    return_scores: bool = False,
) -> Tuple[List[int], Optional[List[float]]]:
    """Select `top_k` candidate indices using Maximal Marginal Relevance (MMR).

    Args:
      query_emb: shape (d,) or (1,d) -- the query embedding vector.
      candidate_embs: shape (n_candidates, d) -- candidate embeddings.
      candidates: list of candidate metadata (only used for length checks).
      top_k: number of items to select.
      diversity: lambda in [0,1]. Higher -> favor relevance; lower -> favor diversity.
                 (we use `diversity` as lambda in the classic MMR formula)
      seed: random seed for tie-breaking.
      return_scores: if True, returns a parallel list of final MMR scores.

    Returns:
      (selected_indices, scores_if_requested)
    """
    rng = np.random.RandomState(seed)

    n = candidate_embs.shape[0]
    if top_k <= 0:
        return ([], [] if return_scores else [])
    if top_k >= n:
        indices = list(range(n))
        return (indices, None) if not return_scores else (indices, [1.0] * n)

    q = query_emb.reshape(1, -1) if query_emb.ndim == 1 else query_emb
    sims_to_query = _cosine_similarity_matrix(candidate_embs, q).squeeze(axis=1)

    # precompute pairwise candidate similarities
    pairwise_sim = _cosine_similarity_matrix(candidate_embs, candidate_embs)

    selected = []
    selected_scores = []

    # first pick: highest similarity to query (tie-break with small noise)
    first_idx_candidates = np.where(sims_to_query == sims_to_query.max())[0]
    if len(first_idx_candidates) > 1:
        first_idx = rng.choice(first_idx_candidates)
    else:
        first_idx = int(first_idx_candidates[0])

    selected.append(first_idx)
    selected_scores.append(float(sims_to_query[first_idx]))

    # MMR loop
    while len(selected) < top_k:
        remaining = [i for i in range(n) if i not in selected]
        mmr_scores = []
        for idx in remaining:
            relevance = sims_to_query[idx]
            max_sim_to_selected = max(pairwise_sim[idx, j] for j in selected)
            mmr_score = diversity * relevance - (1.0 - diversity) * max_sim_to_selected
            mmr_scores.append(mmr_score)

        mmr_scores = np.array(mmr_scores)
        # tie-breaking with small noise
        mmr_scores = mmr_scores + 1e-12 * rng.rand(*mmr_scores.shape)
        chosen_pos = int(np.argmax(mmr_scores))
        chosen_idx = remaining[chosen_pos]
        selected.append(chosen_idx)
        selected_scores.append(float(mmr_scores[chosen_pos]))

    if return_scores:
        return selected, selected_scores
    return selected, None


# ----------------------
# Stratified sampling by metadata (chapter_weights)
# ----------------------


def stratified_sample_by_metadata(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: List[dict],
    metadata: List[Dict],
    chapter_weights: Dict[str, float],
    n_samples: int,
    diversity: float = 0.7,
    seed: Optional[int] = None,
) -> List[int]:
    """Sample `n_samples` candidate indices stratified by `chapter_weights`.

    Args:
      query_emb: embedding for the slot objective.
      candidate_embs: np.ndarray of shape (n_candidates, d).
      candidates: list of candidate objects or ids, same length as candidate_embs.
      metadata: list of metadata dicts aligned with candidates. Expected to contain
                a key like 'chapter' indicating which chapter the snippet belongs to.
      chapter_weights: mapping chapter -> weight (relative, not necessarily sum 1).
      n_samples: total number of items to select.
      diversity: MMR diversity parameter passed to mmr_rerank.
      seed: int seed for reproducibility.

    Returns: list of selected candidate indices (length <= n_samples).
    """
    rng = np.random.RandomState(seed)
    n_candidates = candidate_embs.shape[0]

    # sanitize inputs
    if n_candidates == 0:
        return []
    if len(metadata) != n_candidates:
        raise ValueError("metadata must be same length as candidate_embs")

    # Normalize chapter_weights to counts
    chapters = list(chapter_weights.keys())
    weights = np.array([float(chapter_weights[ch]) for ch in chapters], dtype=float)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        # fallback: uniform
        weights = np.ones_like(weights)
        weights_sum = weights.sum()

    # compute desired counts (floor, then distribute remainder)
    raw_counts = (weights / weights_sum) * n_samples
    base_counts = np.floor(raw_counts).astype(int)
    remainder = int(n_samples - base_counts.sum())

    # distribute remainder to largest fractional parts
    fractions = raw_counts - base_counts
    if remainder > 0:
        order = np.argsort(-fractions)
        for i in range(remainder):
            base_counts[order[i]] += 1

    # collect indices by chapter
    chapter_to_indices = {ch: [] for ch in chapters}
    for idx, md in enumerate(metadata):
        ch = md.get("chapter") or md.get("chap") or md.get("chapter_name")
        # normalize None -> str
        if ch is None:
            continue
        if ch in chapter_to_indices:
            chapter_to_indices[ch].append(idx)

    selected_indices = []

    # For each chapter, run MMR locally to pick required count
    for ch_idx, ch in enumerate(chapters):
        need = int(base_counts[ch_idx])
        pool = chapter_to_indices.get(ch, [])
        if need <= 0 or len(pool) == 0:
            continue

        # Prepare local arrays
        local_embs = candidate_embs[pool]
        # run mmr with top_k = min(need, len(pool))
        pick_k = min(need, len(pool))
        # mmr expects full candidate list; we pass local arrays and map back indices
        selected_local, _ = mmr_rerank(
            query_emb=query_emb,
            candidate_embs=local_embs,
            candidates=[candidates[i] for i in pool],
            top_k=pick_k,
            diversity=diversity,
            seed=seed,
            return_scores=False,
        )
        # map local picks to global indices
        for li in selected_local:
            selected_indices.append(pool[li])

    # If we still need more (due to missing chapters or insufficient pool), fill from global pool
    if len(selected_indices) < n_samples:
        remaining_needed = n_samples - len(selected_indices)
        remaining_pool = [i for i in range(n_candidates) if i not in selected_indices]
        if remaining_pool:
            local_embs = candidate_embs[remaining_pool]
            pick_k = min(remaining_needed, len(remaining_pool))
            picks, _ = mmr_rerank(
                query_emb=query_emb,
                candidate_embs=local_embs,
                candidates=[candidates[i] for i in remaining_pool],
                top_k=pick_k,
                diversity=diversity,
                seed=seed,
                return_scores=False,
            )
            for p in picks:
                selected_indices.append(remaining_pool[p])

    # final truncation (in case of over-select)
    return selected_indices[:n_samples]


# ----------------------
# Convenience wrapper
# ----------------------


def mmr_and_stratified_sample(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: List[dict],
    metadata: List[Dict],
    n_samples: int,
    chapter_weights: Optional[Dict[str, float]] = None,
    diversity: float = 0.7,
    seed: Optional[int] = None,
) -> List[int]:
    """Top-level helper: if chapter_weights provided, do stratified_sample_by_metadata,
    otherwise do global mmr_rerank to pick n_samples.
    """
    if chapter_weights:
        return stratified_sample_by_metadata(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            candidates=candidates,
            metadata=metadata,
            chapter_weights=chapter_weights,
            n_samples=n_samples,
            diversity=diversity,
            seed=seed,
        )
    else:
        picks, _ = mmr_rerank(
            query_emb=query_emb,
            candidate_embs=candidate_embs,
            candidates=candidates,
            top_k=n_samples,
            diversity=diversity,
            seed=seed,
            return_scores=False,
        )
        return picks


# ----------------------
# Simple sanity check (only runs when executed directly)
# ----------------------
if __name__ == "__main__":
    # tiny synthetic test
    rng = np.random.RandomState(42)
    # 10 candidates, embedding dim 8
    cand_embs = rng.randn(10, 8)
    # normalize
    cand_embs = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
    query = rng.randn(8)
    query = query / np.linalg.norm(query)

    candidates = [{"id": f"c{i}"} for i in range(10)]
    metadata = [{"chapter": "Chap1" if i < 6 else "Chap2"} for i in range(10)]
    chapter_weights = {"Chap1": 60, "Chap2": 40}

    picks = mmr_and_stratified_sample(
        query,
        cand_embs,
        candidates,
        metadata,
        n_samples=5,
        chapter_weights=chapter_weights,
        seed=123,
    )
    print("Selected indices:", picks)
    print("Selected ids:", [candidates[i]["id"] for i in picks])
