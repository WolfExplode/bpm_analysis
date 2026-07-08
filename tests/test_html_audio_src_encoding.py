"""Audio filename -> HTML audioSources encoding (assets/interactive_plot.js consumer).

plotting.py builds config_payload["audioSources"] via urllib.parse.quote(audio_src)
(see HtmlReportMixin._build_html_report / plotting.py ~line 1947), and
assets/interactive_plot.js assigns that value directly to `audio.src` and, since the
file:// audio-detection bug fix, also passes it straight to fetch() with no
decodeURIComponent step. Both consumers require the encoded string to be a faithful,
round-trippable percent-encoding of the original filename - these tests guard that for
filenames with brackets, commas, combining marks, and astral-plane / emoji characters
like the ones that triggered "Cannot access audio file: TypeError: Failed to fetch".
"""
import urllib.parse

import pytest

EXOTIC_NAMES = [
    "[05-19-2026] Fainted during a breath hold 𐔌՞. .՞𐦯 [audio only]⭐ [152,90-162bpm].wav",
    "plain_test.wav",
    "spaces and, commas.wav",
    "emoji ⭐🎵 name.wav",
    "brackets [1] (2) {3}.wav",
]


@pytest.mark.parametrize("name", EXOTIC_NAMES)
def test_quote_round_trips_exactly(name):
    quoted = urllib.parse.quote(name)
    assert urllib.parse.unquote(quoted) == name


@pytest.mark.parametrize("name", EXOTIC_NAMES)
def test_quoted_form_has_no_raw_special_characters(name):
    # audio.src and fetch() both consume this string directly as a URL path segment;
    # spaces/brackets/commas/non-ASCII must be percent-encoded, not passed through raw,
    # since relying on the browser to auto-encode an already-decoded string is what
    # caused the original bug.
    quoted = urllib.parse.quote(name)
    for ch in (" ", "[", "]", ",", "{", "}"):
        assert ch not in quoted
    assert quoted.isascii()
