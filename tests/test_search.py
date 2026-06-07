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


def test_title_boost_scales_with_keyword_count(monkeypatch):
    """Two matching keywords must boost by 2x the per-keyword unit (not collapse
    to a boolean). Catches a regression replacing len(matches) with `1 if any`."""
    # Base low enough that base + 2*0.1 stays under the 1.0 clamp.
    raw = [_raw_result("v2", "Nanite and Lumen Deep Dive", score=0.3)]
    monkeypatch.setattr(search, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search, "vector_search", lambda emb, top_k=10: list(raw))

    results = search.search_videos("nanite lumen rendering", top_k=10)
    assert len(results) == 1
    # 'nanite' and 'lumen' are both whole tokens in the title -> 2 keyword boosts.
    assert results[0]["score"] == 0.3 + 2 * search._TITLE_BOOST_PER_KEYWORD


def test_title_boost_hyphen_space_insensitive(monkeypatch):
    """A space-separated query must boost a hyphenated-compound title and vice
    versa: 'real time' should match a 'Real-Time ...' title."""
    raw = [_raw_result("vh", "Real-Time Global Illumination", score=0.3)]
    monkeypatch.setattr(search, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search, "vector_search", lambda emb, top_k=10: list(raw))

    results = search.search_videos("real time rendering", top_k=10)
    assert len(results) == 1
    # 'real' and 'time' (from the split of 'real-time') both match -> 2 boosts.
    assert results[0]["score"] == 0.3 + 2 * search._TITLE_BOOST_PER_KEYWORD


def test_expand_hyphenated_keeps_compound_and_parts():
    assert search._expand_hyphenated({"real-time"}) == {"real-time", "real", "time"}
    assert search._expand_hyphenated({"plain"}) == {"plain"}


def test_seconds_to_hms_rounds_and_clamps():
    """Round to nearest second (not truncate) and clamp negatives to 0."""
    assert search._seconds_to_hms(89.7) == "1:30"
    assert search._seconds_to_hms(0) == "0:00"
    assert search._seconds_to_hms(3600) == "1:00:00"
    assert search._seconds_to_hms(3661.4) == "1:01:01"
    assert search._seconds_to_hms(-5) == "0:00"
    # 59.6 must roll over to 1:00, never ':60'.
    assert search._seconds_to_hms(59.6) == "1:00"
