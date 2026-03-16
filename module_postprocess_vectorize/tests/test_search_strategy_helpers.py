from __future__ import annotations

from module_postprocess_vectorize.postprocess.search import (
    _build_refine_candidates,
    _params_key,
    _sample_evenly,
)


def test_sample_evenly_is_deterministic_and_respects_limit() -> None:
    items = [{"x": i} for i in range(50)]
    sampled = _sample_evenly(items, 7)
    assert len(sampled) == 7
    assert sampled[0]["x"] == 0
    assert sampled[-1]["x"] == 49


def test_refine_candidates_expand_neighbors_without_duplicates() -> None:
    values_map = {
        "extent_thr": [0.45, 0.50, 0.55],
        "boundary_thr": [0.35, 0.45, 0.55],
    }
    top_params = [{"extent_thr": 0.50, "boundary_thr": 0.45}]
    known = {_params_key(top_params[0])}

    cands = _build_refine_candidates(
        top_params=top_params,
        values_map=values_map,
        refine_neighbor_span=1,
        known=known,
    )

    keys = {_params_key(c) for c in cands}
    assert len(cands) == len(keys)
    assert len(cands) >= 2
