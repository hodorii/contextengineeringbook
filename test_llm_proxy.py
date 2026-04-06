import pytest
from unittest.mock import Mock, patch
from llm_proxy import get_ai_response


class MockResponse:
    def __init__(self, content):
        self.content = content


class TestGetAIResponse:
    @patch("llm_proxy.llm")
    def test_string_response(self, mock_llm):
        mock_llm.invoke.return_value = MockResponse("테스트 응답")
        result = get_ai_response("테스트 프롬프트")
        assert result == "테스트 응답"

    @patch("llm_proxy.llm")
    def test_list_response(self, mock_llm):
        mock_llm.invoke.return_value = MockResponse(["첫 번째", "두 번째"])
        result = get_ai_response("테스트 프롬프트")
        assert result == "첫 번째두 번째"

    @patch("llm_proxy.llm")
    def test_dict_response(self, mock_llm):
        mock_llm.invoke.return_value = MockResponse(
            [{"text": "텍스트1"}, {"text": "텍스트2"}]
        )
        result = get_ai_response("테스트 프롬프트")
        assert result == "텍스트1텍스트2"

    @patch("llm_proxy.llm")
    def test_exception_handling(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("API 오류")
        result = get_ai_response("테스트 프롬프트")
        assert "API 호출 중 오류 발생" in result

    @patch("llm_proxy.llm")
    def test_custom_system_instructions(self, mock_llm):
        mock_llm.invoke.return_value = MockResponse("응답")
        custom_instructions = "커스텀 시스템 지시사항"
        result = get_ai_response("프롬프트", system_instructions=custom_instructions)
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert any(msg.content == custom_instructions for msg in call_args)
