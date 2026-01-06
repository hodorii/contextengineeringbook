import os
import logging
# [수정] LangChain의 ChatOpenAI 클래스를 임포트합니다.
from langchain_openai import ChatOpenAI 
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI

# --- 0. 환경 설정: API 키 ---
# 실행 전에 실제 OpenAI API 키를 입력하거나 환경 변수에 설정해야 합니다.
# 예: os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# [수정] .env 파일에서 키를 로드하는 것이 더 안전합니다.
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

# [수정] LangChain의 ChatOpenAI 클래스를 LLM(gpt-5.1)으로 초기화합니다.
llm = ChatOpenAI(
    model="gpt-5.1",
    reasoning={ "effort": "low" }
)

# CLAUDE: 클라이언트 초기화 방식이 다릅니다.
# CLAUDE: llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.1)

# GEMINI: 클라이언트가 아닌, 라이브러리 자체를 설정(configure)합니다.
# GEMINI: import google.generativeai as genai
# GEMINI: genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY"))
# GEMINI: llm = ChatGoogleGenerativeAI(model="gemini-3.0", temperature=0.1)


def get_ai_response(prompt: str) -> str:
    """
    LangChain 챗 모델에 프롬프트를 전달하고, 응답을 텍스트로 반환합니다.
    (Claude/Gemini 사용 시 이 함수 내부의 LLM 객체만 교체하면 됩니다.)
    """
    try:
        # --- LangChain (표준) ---
        # [수정] LangChain의 .invoke() 메서드를 사용합니다.
        #       입력은 OpenAI API와 동일한 'messages' 리스트 형식을 사용합니다.
        response = llm.invoke([{"role": "user", "content": prompt}])
        # [수정] LangChain의 응답 객체에서 .content 속성으로 텍스트를 추출합니다.
        # return response.content.strip()
        if isinstance(response.content, list):
            # 리스트 내부의 각 블록이 텍스트(문자열)라면 모두 합칩니다.
            # (만약 딕셔너리 형태라면 block['text'] 등으로 접근해야 할 수도 있습니다. 
            #  일반적으로는 문자열이거나 text 속성을 가진 객체일 것입니다.)
            return "".join([str(block) for block in response.content]).strip()
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

# --- 1. 컨텍스트가 없는 경우 ---
print("="*50)
print("--- [사례 1] 컨텍스트 없는 요청 (The Problem) ---")
print("="*50)

# AI는 우리 회사의 코딩 규칙(snake_case, logging, db 객체)을 전혀 모릅니다.
prompt_without_context = "데이터베이스에서 사용자 ID로 사용자 정보를 가져오는 파이썬 함수를 만들어줘."

print(f"[요청 프롬프트]\n{prompt_without_context}\n")

# CLAUDE/GEMINI: 이 아랫부분은 'get_ai_response' 함수가
# CLAUDE/GEMINI: LangChain의 표준 인터페이스(.invoke)를 사용하므로 변경할 필요가 없습니다.
ai_response_1 = get_ai_response(prompt_without_context)
print(f"[AI의 응답 (컨텍L스트 X)]\n{ai_response_1}")
