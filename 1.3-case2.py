import os
import logging
# [수정] LangChain의 ChatOpenAI 클래스를 임포트합니다.
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI 
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI

# [수정] .env 파일 로드를 위한 라이브러리 임포트
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 ---
# [수정] .env 파일에서 키를 로드하고 환경 변수로 설정합니다.
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

# [수정] LangChain의 ChatOpenAI 클래스를 LLM(두뇌)으로 초기화합니다.
# 요청하신대로 model="gpt-4o"로 변경합니다.
llm = ChatOpenAI(
    model=”gpt-5.1”,
    reasoning={ "effort": "low" }
)
# CLAUDE: 클라이언트 초기화 방식이 다릅니다. (LangChain 표준)
# CLAUDE: os.environ[“ANTHROPIC_API_KEY”] = os.getenv(“ANTHROPIC_API_KEY”, “YOUR_
CLAUDE_KEY”)
# CLAUDE: llm = ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0.1)

# GEMINI: 클라이언트가 아닌 라이브러리 자체를 설정합니다. (LangChain 표준)
# GEMINI: import google.generativeai as genai
# GEMINI: os.environ[“GOOGLE_API_KEY”] = os.getenv(“GOOGLE_API_KEY”, “YOUR_GEMINI_
KEY”)
# GEMINI: genai.configure(api_key=os.environ[“GOOGLE_API_KEY”])
# GEMINI: llm = ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0.1)


# --- 1. AI 응답 생성 함수 정의 ---
def get_ai_response(prompt: str) -> str:
    """
    [수정] 컨텍스트로 보강된(Augmented) 프롬프트를 LangChain 챗 모델에 전달하여
    최종 응답을 생성(Generation)합니다.
    (Claude/Gemini 사용 시 이 함수 내부의 LLM 객체만 교체하면 됩니다.)
    """
    print("[AI 모델에 응답 요청 중...]\n")
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

# --- 2. 컨텍스트 엔지니어링: 프롬프트 템플릿 설계 ---
# [수정] 원본 코드에 누락되었던 템플릿 정의를 추가합니다.
base_prompt_template = """
당신은 AI 법률 자문가입니다.
아래에 제공된 [컨텍스트]를 바탕으로, [질문]에 대한 법적 위험을 분석하세요.

- 만약 [컨텍스트]에 "제공된 컨텍스트 없음"이라고 되어있다면, 계약서의 '일반적인' 법적 위험 4가지를 원론적으로 설명하세요.
- 만약 [컨텍스트]에 실제 계약서 내용이 있다면, 해당 내용에서 발견되는 '구체적인' 법적 위험 2가지를 짚어서 분석하세요.

[컨텍스트]
{context}

[질문]
{question}
"""

# --- 3. 시나리오별 실행 ---

question = "이 계약서의 법적 위험은 무엇인가?"

context_2 = """
[소프트웨어 개발 계약서 초안]
제 5조 (지식재산권) 개발 과정에서 발생하는 모든 IP는 "을"에게 귀속된다.
제 8조 (검수 기준) "갑"은 "을"이 제공하는 최종 산출물을 검수한다.
"""
prompt_2 = base_prompt_template.format(context=context_2, question=question)

print(f"[전송된 프롬프트]:\n{prompt_2}\n")
response_2 = get_ai_response(prompt_2)
print(f"[AI의 구체적인 답변]:\n{response_2}\n")

# --- [사례 3] 부동산 임대차 계약서 컨텍스트 제공 ---
print("="*50)
print("--- [사례 3] 부동산 임대차 컨텍스트가 있는 경우 ---")
print("="*50)

context_3 = """
[부동산 임대차 계약서 초안]
제 3조 (수선 의무) 임차인의 고의, 과실이 아닌 '주요 설비'의 노후화로 인한 고장은 임대인이 수리한다.
(특약사항 없음)
"""
prompt_3 = base_prompt_template.format(context=context_3, question=question)

print(f"[전송된 프롬프트]:\n{prompt_3}\n")
response_3 = get_ai_response(prompt_3)
print(f"[AI의 구체적인 답변]:\n{response_3}\n")
