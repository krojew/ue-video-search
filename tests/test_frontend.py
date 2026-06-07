"""Lock-in tests for the static frontend (static/index.html).

There is no JS test runner in this project, so these are lightweight static
assertions over the served file. They guard the XSS / SSE / search-race fixes
against an accidental revert (e.g. dropping escAttr from an attribute, or
removing the JSON.parse try/catch). They are intentionally coarse — a real
DOM/jsdom suite would be better, but this at least fails loudly on a regression.
"""
from __future__ import annotations

import re
import shutil
import subprocess
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


def test_attr_escape_map_has_all_five_entities():
    """ATTR_ESCAPE_MAP must map each of & < > \" ' to its entity.

    Asserting the MAP (not just the regex character class) closes the hole
    where deleting e.g. `'"': '&quot;'` would leave escAttr('"') === 'undefined'
    — a real XSS regression — while a regex-only check still passed.
    """
    m = re.search(r"const ATTR_ESCAPE_MAP\s*=\s*\{(.*?)\}", _HTML, re.DOTALL)
    assert m, "ATTR_ESCAPE_MAP not found"
    body = m.group(1)
    for pair in ["'&': '&amp;'", "'<': '&lt;'", "'>': '&gt;'",
                 "'\"': '&quot;'", "\"'\": '&#39;'"]:
        assert pair in body, f"ATTR_ESCAPE_MAP missing mapping {pair}"


def test_escattr_behaviour_via_node():
    """Execute escAttr in node and assert it neutralizes an attribute breakout.

    This is the behavioral guard: it catches a broken/missing map entry that a
    static check could miss. Skips cleanly if node is unavailable.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    map_m = re.search(r"const ATTR_ESCAPE_MAP\s*=\s*\{.*?\};", _HTML, re.DOTALL)
    fn_m = re.search(r"function escAttr\(s\)\s*\{.*?\n  \}", _HTML, re.DOTALL)
    assert map_m and fn_m, "could not extract escAttr + map from index.html"

    script = (
        map_m.group(0)
        + "\n"
        + fn_m.group(0)
        + "\nconst payload = 'a\"><img src=x onerror=alert(1)>';"
        + "\nconst out = escAttr(payload);"
        + "\nif (/[<>\"]/.test(out)) { console.error('UNESCAPED: ' + out); process.exit(1); }"
        + "\nif (out.indexOf('&quot;') === -1 || out.indexOf('&lt;') === -1) "
        + "{ console.error('MISSING ENTITY: ' + out); process.exit(1); }"
        + "\nconsole.log(out);"
    )
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"escAttr failed to escape: {proc.stdout}{proc.stderr}"


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
