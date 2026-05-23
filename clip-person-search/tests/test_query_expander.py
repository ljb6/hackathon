import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
import query_expander


def _mock_response(text):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def test_expand_returns_llm_output():
    with patch.object(query_expander._client, "messages") as mock_msgs:
        mock_msgs.create.return_value = _mock_response("a person riding a bicycle")
        result = query_expander.expand_query("bike")
    assert result == "a person riding a bicycle"


def test_empty_string_returns_unchanged():
    with patch.object(query_expander._client, "messages") as mock_msgs:
        result = query_expander.expand_query("   ")
    mock_msgs.create.assert_not_called()
    assert result == "   "


def test_api_failure_returns_raw():
    with patch.object(query_expander._client, "messages") as mock_msgs:
        mock_msgs.create.side_effect = Exception("network error")
        result = query_expander.expand_query("running")
    assert result == "running"
