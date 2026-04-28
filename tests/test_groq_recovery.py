"""
tests/test_groq_recovery.py
===========================
Tests for I15 — Centralised Groq XML tool-call recovery helper.

Covers ``src.agents.groq_recovery.parse_groq_xml_tool_call`` across:

* well-formed JSON-body tool calls (the common case)
* the legacy ``search_term="value"`` kwarg-style body
* propose_state_change variant with full JSON kwargs
* malformed and empty inputs (must return None, never raise)
* unrelated error strings (no XML tag at all → None)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.groq_recovery import parse_groq_xml_tool_call


# ---------------------------------------------------------------------------
# Happy-path: well-formed XML wrappers around JSON args
# ---------------------------------------------------------------------------

def test_parses_query_data_with_json_body():
    err = (
        'Failed to call a function. Please adjust your prompt. See '
        '\'failed_generation\' for more details. Tool call: '
        '<function=query_data>{"search_term": "ITEM-PLASTIC-01"}</function>'
    )
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "query_data"
    assert result["arguments"] == {"search_term": "ITEM-PLASTIC-01"}


def test_parses_propose_state_change_with_full_json_body():
    err = (
        'Tool call format error: '
        '<function=propose_state_change>'
        '{"row_key": "ITEM-PLASTIC-01", "target_column": "current_stock", '
        '"delta": -500, "justification": "buffer reduction"}'
        '</function>'
    )
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "propose_state_change"
    args = result["arguments"]
    assert args["row_key"] == "ITEM-PLASTIC-01"
    assert args["target_column"] == "current_stock"
    assert args["delta"] == -500
    assert args["justification"] == "buffer reduction"


def test_parses_get_dataset_schema_with_empty_body():
    err = '<function=get_dataset_schema></function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "get_dataset_schema"
    assert result["arguments"] == {}


def test_parses_function_with_attributes_in_open_tag():
    err = '<function=query_data extra="ignored">{"search_term": "brake"}</function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "query_data"
    assert result["arguments"] == {"search_term": "brake"}


# ---------------------------------------------------------------------------
# Kwarg-style fallback: search_term="value"
# ---------------------------------------------------------------------------

def test_parses_kwarg_style_search_term():
    err = '<function=query_data>search_term="SFT-HD"</function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "query_data"
    assert result["arguments"] == {"search_term": "SFT-HD"}


def test_parses_kwarg_style_with_single_quotes():
    err = "<function=query_data>search_term='Insulin'</function>"
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["arguments"] == {"search_term": "Insulin"}


def test_parses_bare_string_body_as_search_term():
    err = '<function=query_data>plastic</function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    # Bare string fallback should hand the body off as a search_term
    # so query_data still has a chance of producing a useful match.
    assert result["arguments"] == {"search_term": "plastic"}


# ---------------------------------------------------------------------------
# Failure modes: must return None, never raise
# ---------------------------------------------------------------------------

def test_empty_string_returns_none():
    assert parse_groq_xml_tool_call("") is None


def test_none_input_returns_none():
    # Defensive: callers may pass str(None) or accidentally None.
    assert parse_groq_xml_tool_call(None) is None  # type: ignore[arg-type]


def test_unrelated_error_returns_none():
    err = "ConnectionError: failed to reach Groq endpoint after 3 retries"
    assert parse_groq_xml_tool_call(err) is None


def test_malformed_xml_no_function_tag_returns_none():
    err = "<bogus>not a function call</bogus>"
    assert parse_groq_xml_tool_call(err) is None


def test_malformed_xml_missing_close_tag_returns_none():
    # No closing </function> — regex must not match a partial tag.
    err = '<function=query_data>{"search_term": "x"}'
    assert parse_groq_xml_tool_call(err) is None


def test_malformed_xml_does_not_raise():
    # Pathological input should never crash the parser.
    pathological_inputs = [
        "<function=>",
        "<function=name<<<>>>",
        "</function>",
        "<<>>",
        "function=query_data",
        "<function=query_data><function=>",
    ]
    for err in pathological_inputs:
        # Either None or a valid dict — never an exception.
        result = parse_groq_xml_tool_call(err)
        assert result is None or (
            isinstance(result, dict)
            and "tool_name" in result
            and "arguments" in result
            and isinstance(result["arguments"], dict)
        )


# ---------------------------------------------------------------------------
# Shape contract: never returns non-dict arguments
# ---------------------------------------------------------------------------

def test_arguments_is_always_dict_for_propose_state_change():
    # JSON array body — would parse to list, not dict. The helper
    # should still return a dict (empty) so propose_state_change can
    # produce its own helpful error.
    err = '<function=propose_state_change>[1, 2, 3]</function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert result["tool_name"] == "propose_state_change"
    assert isinstance(result["arguments"], dict)


def test_returned_dict_has_exactly_two_keys():
    err = '<function=query_data>{"search_term": "x"}</function>'
    result = parse_groq_xml_tool_call(err)
    assert result is not None
    assert set(result.keys()) == {"tool_name", "arguments"}
