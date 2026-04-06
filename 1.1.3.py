import os
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic

# .env 파일 로드를 위한 라이브러리 임포트
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 ---
# .env 파일에서 키를 로드하고 환경 변수로 설정합니다.
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# API 키 미설정 시 테스트(모의) 모드로 동작합니다.
API_KEY = os.getenv("GOOGLE_API_KEY")

# LangChain의 ChatGoogleGenerativeAI 클래스를 LLM(두뇌)으로 초기화합니다.
llm = None
llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.7, max_output_tokens=2000
    )

# OpenAI: llm = ChatOpenAI(model="gpt-5.1", reasoning={ "effort": "low" }, max_tokens=2000)
# CLAUDE: llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)


# --- 1. AI 모델과 상호작용하는 함수 정의 ---
def get_ai_response(messages: list) -> str:
    """
    [수정] 주어진 대화 기록(messages)을 바탕으로 LangChain 챗 모델의 응답을 반환합니다.
    대화 기록이 컨텍스트 역할을 합니다.
    """
    try:
        # --- LangChain (표준) ---
        # [수정] LangChain의 .invoke() 메서드를 사용합니다.
        #       입력은 OpenAI API와 동일한 'messages' 리스트 형식을 사용합니다.
        response = llm.invoke(messages)
        # [수정] LangChain의 응답 객체에서 .content 속성으로 텍스트를 추출합니다.
        # return response.content.strip()
        if isinstance(response.content, list):
            # 리스트 일 경우
            # return "".join([str(block) for block in response.content]).strip()
            # 딕셔너리 구조일 경우 (예상)
            return "".join(
                [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response.content
                ]
            ).strip()
        else:
            # 문자열인 경우 기존대로 처리
            return response.content.strip()

        # --- CLAUDE로 변경 시 (위 llm 객체만 변경하면, 이 함수는 수정 불필요) ---
        # CLAUDE: response = llm.invoke(messages)
        # CLAUDE: return response.content.strip()

        # --- GEMINI로 변경 시 (위 llm 객체만 변경하면, 이 함수는 수정 불필요) ---
        # GEMINI: response = llm.invoke(messages)
        # GEMINI: return response.content.strip()

    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"


# --- 2. 컨텍스트가 없는 경우 (AI의 단기 기억상실증 발생) ---
print("--- 컨텍스트가 없는 대화 ---")

# 첫 번째 질문 (독립적인 대화 시작)
question_1 = "일론 머스크는 누구야?"
# 대화 기록 없이, 첫 질문만 전달
messages_1 = [{"role": "user", "content": question_1}]
answer_1 = get_ai_response(messages_1)
print(f"사용자: {question_1}")
print(f"AI: {answer_1}")

print("-" * 20)

# 두 번째 질문 (완전히 새로운 대화로 취급)
question_2 = "그의 어머니는 어떤 분이야?"
# 이전 대화 기록 없이, 두 번째 질문만 전달
# AI는 '그'가 '일론 머스크'임을 알 수 있는 컨텍스트가 없음
messages_2 = [{"role": "user", "content": question_2}]
answer_2 = get_ai_response(messages_2)
print(f"사용자: {question_2}")
print(f"AI: {answer_2}\n")  # AI는 '그'가 누구인지 되물을 가능성이 높음

# --- 3. 컨텍스트를 제공하는 경우 (대화 맥락 유지) ---
print("--- 컨텍스트를 제공하는 대화 ---")

# 첫 번째 질문과 응답을 포함한 대화 기록 생성
conversation_history = [
    {"role": "user", "content": question_1},
    {"role": "assistant", "content": answer_1},  # 첫 번째 응답을 기록에 추가
    {"role": "user", "content": question_2},  # 두 번째 질문 추가
]
# CLAUDE/GEMINI: 이 'conversation_history' 리스트를 만드는 로직은
# CLAUDE/GEMINI: 플랫폼에 상관없이 동일하게 유지됩니다.
# CLAUDE/GEMINI: 이 리스트가 get_ai_response 함수로 전달되어,
# CLAUDE/GEMINI: 함수 내부에서 각 플랫폼에 맞게 처리(Gemini의 경우 변환)됩니다.

# 전체 대화 기록을 컨텍스트로 제공하여 두 번째 질문 실행
# AI는 이전 대화를 통해 '그'가 '일론 머스크'임을 추론 가능
answer_3_with_context = get_ai_response(conversation_history)
print(f"사용자: {question_2}")
print(f"AI: {answer_3_with_context}")  # AI가 메이 머스크에 대해 답변할 가능성이 높음
