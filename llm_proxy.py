import os
from typing import Optional, Any, List
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_INSTRUCTIONS = """
## ROLE 
Python Expert
## RULE
- 한국어 기반.(주석 포함)
- SRP
- Python Naming Convention 준수
- Naming Convention은 가독성 핵심
- 모르는 것, 헷갈리는 것 사용자에게 다시 확인
- 참조한 것은 반드시 링크 남긴다
"""


class EmbeddingProxy:
    """임베딩 전용 프록시 - LLM과 독립적으로 선택 가능"""

    PROVIDERS = {
        "huggingface": "HuggingFace (sentence-transformers/all-MiniLM-L6-v2)",
        "openai": "OpenAI (text-embedding-3-small)",
        "google": "Google Gemini (gemini-embedding-001)",
        "ollama": "Ollama (nomic-embed-text)",
    }

    def __init__(self, provider: str = "huggingface", model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.embeddings = self._create_embeddings()

    def _create_embeddings(self):
        """공급자에 따라 임베딩 인스턴스 생성"""
        if self.provider == "google":
            return self._create_google_embeddings()
        elif self.provider == "ollama":
            return self._create_ollama_embeddings()
        elif self.provider == "openai":
            return self._create_openai_embeddings()
        elif self.provider == "huggingface":
            return self._create_huggingface_embeddings()
        else:
            raise ValueError(f"지원하지 않는 임베딩 공급자: {self.provider}")

    def _create_google_embeddings(self):
        class GoogleEmbedding:
            """Google Gemini 임베딩 래퍼"""

            def __init__(self, api_key: str):
                self.api_key = api_key
                self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

            def _embed(self, text: str) -> List[float]:
                import requests

                response = requests.post(
                    f"{self.url}?key={self.api_key}",
                    json={"content": {"parts": [{"text": text}]}},
                )
                if response.status_code != 200:
                    raise Exception(f"Embedding error: {response.text}")
                return response.json()["embedding"]["values"]

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self._embed(t) for t in texts]

            def embed_query(self, text: str) -> List[float]:
                return self._embed(text)

            def __call__(self, text: str) -> List[float]:
                return self._embed(text)

        return GoogleEmbedding(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        )

    def _create_ollama_embeddings(self):
        try:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=self.model_name or "nomic-embed-text")
        except ImportError:
            raise ImportError(
                "langchain-ollama 설치 필요: pip install langchain-ollama"
            )

    def _create_openai_embeddings(self):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=self.model_name or "text-embedding-3-small")
        except ImportError:
            raise ImportError(
                "langchain-openai 설치 필요: pip install langchain-openai"
            )

    def _create_huggingface_embeddings(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError:
            raise ImportError(
                "langchain-community 설치 필요: pip install langchain-community"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩"""
        return self.embeddings.embed_query(text)

    @classmethod
    def available_providers(cls) -> dict:
        """사용 가능한 공급자 확인"""
        available = {}
        for provider in cls.PROVIDERS:
            try:
                proxy = EmbeddingProxy(provider)
                available[provider] = True
            except Exception:
                available[provider] = False
        return available


class LLMProxy:
    """다양한 LLM 공급자를 지원하는 프록시 클래스 - 임베딩 제외"""

    PROVIDERS = {
        "google": "Google Gemini",
        "ollama": "Ollama (로컬)",
        "openai": "OpenAI",
        "anthropic": "Anthropic Claude",
    }

    def __init__(self, provider: str = "google", model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.llm = self._create_llm()

    def _create_llm(self):
        """공급자에 따라 LLM 인스턴스 생성 (지연 초기화)"""
        if self.provider == "google":
            return self._create_google()
        elif self.provider == "ollama":
            return self._create_ollama()
        elif self.provider == "openai":
            return self._create_openai()
        elif self.provider == "anthropic":
            return self._create_anthropic()
        else:
            raise ValueError(f"지원하지 않는 공급자: {self.provider}")

    def _create_google(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_name or "gemini-3-flash-preview",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.7,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai 설치 필요: pip install langchain-google-genai"
            )

    def _create_ollama(self):
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.model_name or "qwen3:1.7b", temperature=0.7)
        except ImportError:
            raise ImportError(
                "langchain-ollama 설치 필요: pip install langchain-ollama"
            )

    def _create_openai(self):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name or "gpt-4o-mini",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
            )
        except ImportError:
            raise ImportError(
                "langchain-openai 설치 필요: pip install langchain-openai"
            )

    def _create_anthropic(self):
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model_name or "claude-sonnet-4-20250514",
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.7,
            )
        except ImportError:
            raise ImportError(
                "langchain-anthropic 설치 필요: pip install langchain-anthropic"
            )

    def get_response(
        self, prompt: str, system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=system_instructions),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return self._parse_response(response)
        except Exception as e:
            return f"[{self.provider.upper()}] 오류: {e}"

    def _parse_response(self, response) -> str:
        if not hasattr(response, "content"):
            return str(response)

        content = response.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()

        if isinstance(content, dict):
            return content.get("text", "").strip()

        return str(content).strip()

    @classmethod
    def available_providers(cls) -> dict:
        """사용 가능한 공급자 확인"""
        available = {}
        for provider in cls.PROVIDERS:
            try:
                proxy = LLMProxy(provider)
                available[provider] = True
            except Exception:
                available[provider] = False
        return available


def get_ai_response(
    prompt: str, system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS
) -> str:
    """편의 함수 - 기본 Google 사용"""
    proxy = LLMProxy(provider="google")
    return proxy.get_response(prompt, system_instructions)


if __name__ == "__main__":
    print("사용 가능한 LLM 공급자:")
    for provider, available in LLMProxy.available_providers().items():
        status = "✓" if available else "✗"
        print(f"  {status} {provider}: {LLMProxy.PROVIDERS[provider]}")

    # 테스트 (ollama만)
    print("\n--- Testing Ollama ---")
    try:
        proxy = LLMProxy(provider="ollama", model_name="qwen3.5:0.8b")
        print(proxy.get_response("안녕하세요!"))
    except Exception as e:
        print(f"Ollama 오류: {e}")
