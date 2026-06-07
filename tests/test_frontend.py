"""Lock-in tests for the static frontend (static/index.html).

There is no JS test runner in this project, so these are lightweight static
assertions over the served file. They guard the XSS / SSE / search-race fixes
against an accidental revert (e.g. dropping escAttr from an attribute, or
removing the JSON.parse try/catch). They are intentionally coarse — a real
DOM/jsdom suite would be better, but this at least fails loudly on a regression.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text()


@pytest.mark.parametrize(
    "interp",
    [
        "${escAttr(thumbUrl)}",
        "${escAttr(video.video_url)}",
        "${escAttr(seg.timestamped_url)}",
    ],
)
def test_dynamic_attributes_are_escaped(interp):
    """Every dynamic value placed into an HTML attribute must go through escAttr."""
    assert interp in _HTML, f"expected escaped attribute interpolation {interp!r}"


def test_no_raw_dynamic_attribute_interpolation():
    """No bare ${...} immediately inside a src=" / href=" attribute (would be XSS)."""
    raw_attr = re.findall(r'(?:src|href)="\$\{(?!escAttr)', _HTML)
    assert raw_attr == [], f"unescaped attribute interpolation(s): {raw_attr}"


def test_video_id_is_url_encoded_in_thumbnail():
    assert "encodeURIComponent(video.video_id)" in _HTML


def test_escattr_helper_encodes_quotes_and_angles():
    """The escAttr helper body must encode &, <, >, \" and '."""
    # Grab the escAttr function body.
    m = re.search(r"function escAttr\(s\)\s*\{(.*?)\n  \}", _HTML, re.DOTALL)
    assert m, "escAttr helper not found"
    body = m.group(1)
    for ch in ["&", "<", ">", '"', "'"]:
        assert ch in body, f"escAttr does not appear to handle {ch!r}"


def test_sse_status_handler_guards_json_parse():
    """The SSE 'status' handler must wrap JSON.parse in try/catch."""
    # Find the status listener and assert a try/catch surrounds JSON.parse.
    idx = _HTML.find("addEventListener('status'")
    assert idx != -1, "SSE status listener not found"
    snippet = _HTML[idx:idx + 400]
    assert "try" in snippet and "JSON.parse" in snippet and "catch" in snippet


def test_search_has_sequence_guard():
    """doSearch must use a monotonic sequence guard against stale responses."""
    assert "searchSeq" in _HTML
    assert "++searchSeq" in _HTML
