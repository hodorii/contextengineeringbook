import os
import logging
# [수정] LangChain의 ChatOpenAI 클래스를 임포트합니다.
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI

# --- 0. 환경 설정: API 키 ---
# [수정] .env 파일에서 키를 로드하고 환경 변수로 설정하는 것을 권장합니다.
from dotenv import load_dotenv
load_dotenv()
# 실행 전에 실제 OpenAI API 키를 입력하거나 환경 변수에 설정해야 합니다.
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

# [수정] LangChain의 ChatOpenAI 클래스를 LLM으로 초기화합니다.
llm = ChatOpenAI(
    model=”gpt-5.1”,
    reasoning={ "effort": "low" }
)

# CLAUDE: 클라이언트 초기화 방식이 다릅니다.
# CLAUDE: llm = ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0.1,
api_key=os.getenv(“ANTHROPIC_API_KEY”))

# GEMINI: 클라이언트가 아닌 라이브러리 자체를 설정합니다.
# GEMINI: import google.generativeai as genai
# GEMINI: genai.configure(api_key=os.getenv(“GOOGLE_API_KEY”, “YOUR_GEMINI_KEY”))
# GEMINI: llm = ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0.1)


def get_ai_response(prompt: str) -> str:
    """
    [수정] LangChain 챗 모델에 프롬프트를 전달하고, 응답을 텍스트로 반환합니다.
    (Claude/Gemini 사용 시 이 함수 내부의 LLM 객체만 교체하면 됩니다.)
    """
    print("AI 모델에 응답을 요청합니다...")
    try:
        # --- LangChain (표준) ---
        # [수정] LangChain의 .invoke() 메서드를 사용합니다.
        #       입력은 OpenAI API와 동일한 'messages' 리스트 형식을 사용합니다.
        response = llm.invoke([{"role": "user", "content": prompt}])
        # [수정] LangChain의 응답 객체에서 .content 속성으로 텍스트를 추출합니다.
        if isinstance(response.content, list):
            # 리스트 일 경우
            # return "".join([str(block) for block in response.content]).strip()
            # 딕셔너리 구조일 경우 (예상)
            return "".join([
                block.get('text', '') if isinstance(block, dict) else str(block) 
                for block in response.content
            ]).strip()
        else:
            # 문자열인 경우 기존대로 처리
            return response.content.strip()

        # --- CLAUDE로 변경 시 (위 llm 객체만 변경하면, 이 함수는 수정 불필요) ---
        # CLAUDE: response = llm.invoke([{"role": "user", "content": prompt}])
        # CLAUDE: return response.content.strip()

        # --- GEMINI로 변경 시 (위 llm 객체만 변경하면, 이 함수는 수정 불필요) ---
        # GEMINI: response = llm.invoke([{"role": "user", "content": prompt}])
        # GEMINI: return response.content.strip()
        
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# --- 1. 평범한 프롬프트 (컨텍스트 부족) ---
bad_prompt = "파이썬으로 리스트를 정렬하는 함수 알려줘."

print("="*50)
print("--- [사례 1] 평범한 프롬프트 실행 ---")
print("="*50)
print(f"[요청]: {bad_prompt}\n")
response_1 = get_ai_response(bad_prompt)
print(f"[AI 응답 (컨텍스트 X)]:\n{response_1}")

# --- 2. 잘 설계된 프롬프트 (컨텍스트 주입) ---
good_prompt = """
당신은 파이썬 초심자를 가르치는 친절한 코딩 튜터입니다.
아래 조건에 맞춰 파이썬 리스트 정렬 함수를 설명해주세요.

1. 오름차순과 내림차순 정렬 방법을 모두 보여주세요.
2. 각 코드 블록 아래에 초심자가 이해하기 쉽게 상세한 주석을 달아주세요.
3. 원본 리스트를 변경하는 `sort()`와 새 리스트를 반환하는 `sorted()`의 차이점을 표로 비교해주세요.
"""

print("\n" + "="*50)
print("--- [사례 2] 잘 설계된 프롬프트 실행 ---")
print("="*50)
print(f"[요청]: (상세 프롬프트 - 위 코드 참조)\n")
response_2 = get_ai_response(good_prompt)
print(f"[AI의 응답 (컨텍스트 O)]:\n{response_2}")
