"""Hermetic unit tests for src.search (no network: embed/vector_search stubbed)."""

import src.search as search


def _raw_result(video_id, title, score, start=0.0, end=10.0, text="some text"):
    return {
        "score": score,
        "video_id": video_id,
        "video_title": title,
        "video_url": f"https://youtu.be/{video_id}",
        "start": start,
        "end": end,
        "text": text,
    }


def test_title_boost_matches_whole_words_not_substrings(monkeypatch):
    """Query 'AI animation' should boost a title where 'ai' is a whole token,
    but NOT a title where 'ai' only appears inside 'chain'/'detailing'."""
    # Base scores low enough that a single-keyword boost (0.1) stays under 1.0.
    raw = [
        _raw_result("v_boost", "AI for NPC behavior", score=0.5),
        _raw_result("v_nope", "Chain physics detailing", score=0.5),
    ]

    monkeypatch.setattr(search, "embed_query", lambda q: [0.0, 0.0, 0.0])
    monkeypatch.setattr(search, "vector_search", lambda emb, top_k=10: list(raw))

    results = search.search_videos("AI animation", top_k=10)
    by_id = {r["video_url"].rsplit("/", 1)[-1]: r for r in results}

    boosted = by_id["v_boost"]["score"]
    not_boosted = by_id["v_nope"]["score"]

    # 'ai' is a whole token in "AI for NPC behavior" -> boosted by one keyword.
    assert boosted == 0.5 + search._TITLE_BOOST_PER_KEYWORD
    # 'ai' only a substring of 'chain'/'detailing' -> no boost.
    assert not_boosted == 0.5
    assert boosted > not_boosted


def test_seconds_to_hms_rounds_and_clamps():
    """Round to nearest second (not truncate) and clamp negatives to 0."""
    assert search._seconds_to_hms(89.7) == "1:30"
    assert search._seconds_to_hms(0) == "0:00"
    assert search._seconds_to_hms(3600) == "1:00:00"
    assert search._seconds_to_hms(3661.4) == "1:01:01"
    assert search._seconds_to_hms(-5) == "0:00"
