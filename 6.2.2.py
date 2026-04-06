import os
# [수정] LangChain의 ChatOpenAI 클래스를 임포트합니다.
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 로드 ---
load_dotenv()
# OpenAI API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")

# --- 1. LLM(두뇌) 정의 ---
# 학생의 복잡한 학습 이력과 규칙을 정확히 따르도록 gpt-4o와 같은 고성능 모델을 사용합니다.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# GEMINI: llm = ChatGoogleGenerativeAI(model="gemini-3.0", temperature=0.3)

def get_ai_response(prompt: str) -> str:
    """
    최종 프롬프트를 AI 모델(ChatModel)에 전달하고,
    생성된 답변 텍스트를 반환합니다.
    """
    print("\n--- AI 과외 선생님이 맞춤형 학습 계획을 생성 중입니다... ---")
    try:
        # LangChain의 invoke 메서드를 사용하여 AI 호출
        # HumanMessage로 전체 프롬프트를 전달합니다.
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        # return response.content.strip()
        if isinstance(response.content, list):
            # 리스트 내부의 각 블록이 텍스트(문자열)라면 모두 합칩니다.
            # (만약 딕셔너리 형태라면 block['text'] 등으로 접근해야 할 수도 있습니다. 
            #  일반적으로는 문자열이거나 text 속성을 가진 객체일 것입니다.)
            return "".join([str(block) for block in response.content]).strip()
        else:
            # 문자열인 경우 기존대로 처리
            return response.content.strip()
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# --- 2. '이민준' 학생의 학습 데이터를 담은 '개인별 학습 파일' (컨텍스트) ---
student_learning_file = """
# 학생 개인별 학습 파일

## 1. 학생 프로필
- 이름: 이민준
- 학년: 중학교 1학년
- 학습 목표: 1학기 기말고사 수학 90점 이상 달성
- 학습 성향: 시각 자료(도표, 그림) 활용 시 이해도 높음. 문장제 문제 독해에 어려움을 느낌.

## 2. 주요 단원별 학습 이력
- 1단원 (소인수분해): 성취도 상 (95점)
- 2단원 (정수와 유리수): 성취도 중 (85점)
- 3단원 (문자와 식): 성취도 하 (65점) - **집중 학습 필요**

## 3. 핵심 오답 노트 (최근 2주)
- **오답 유형 A: 일차방정식 활용 (문장제)**
  - **문제 예시:** "연속하는 세 홀수의 합이 57일 때, 가장 큰 수는?"
  - **학생의 실수:** 세 홀수를 `x, x+1, x+2`로 잘못 설정하여 식을 세움.
  - **분석된 약점:** 문장을 수학적 식으로 정확하게 변환하는 능력이 부족함. 특히 '연속하는 짝수/홀수'와 같은 특정 조건에 대한 개념 이해가 필요함.

- **오답 유형 B: 음수를 포함한 식의 값 계산**
  - **문제 예시:** `a = -2`일 때, `-a²`의 값을 구하시오.
  - **학생의 실수:** `(-(-2))²`로 계산하여 `4`라고 답함.
  - **분석된 약점:** 거듭제곱과 괄호의 연산 우선순위 규칙을 혼동함. 음수를 대입할 때 괄호를 사용하는 습관이 필요함.
"""

# --- 3. 프롬프트 생성 함수 (제공된 코드) ---
def generate_tutoring_prompt(learning_file, user_request):
    """
    학생 학습 파일과 요청을 결합하여
    AI 과외 선생님에게 전달할 최종 프롬프트를 생성하는 함수.
    """
    prompt = f"""
    당신은 학생 데이터를 기반으로 개인화된 학습 계획을 세우는 최고의 'AI 수학 과외 선생님'입니다.
    당신의 임무는 아래 '[학생 개인별 학습 파일]'을 면밀히 분석하고, 학생의 현재 수준과 약점에 가장 적합한 학습 자료를 생성하는 것입니다.

    **[작업 지시]**
    1.  먼저, '[학생 개인별 학습 파일]'을 분석하여 학생의 가장 시급한 **핵심 약점 2가지**를 진단하세요.
    2.  진단된 각 약점을 극복하는 데 도움이 될 **맞춤형 연습 문제**를 각각 2문제씩, 총 4문제를 출제하세요. 문제에는 반드시 **자세한 풀이 과정과 정답**을 포함해야 합니다.
    3.  마지막으로, 오늘 학습한 내용을 바탕으로 **다음 학습 세션을 위한 간단한 계획**을 제안하세요.
    4.  학생의 학습 성향을 고려하여, 필요하다면 시각적인 비유나 도표를 활용하여 설명하세요.

    ---
    [학생 개인별 학습 파일]

    {learning_file}

    ---

    [학생의 요청]
    {user_request}

    [AI 과외 선생님의 답변]
    """
    return prompt

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    
    # 학생의 요청
    user_request = "선생님, 오늘 저 수학 공부 뭐 해야 할까요? 맞춤 문제 좀 내주세요."

    # 'AI 과외 선생님' 프롬프트 생성
    final_prompt = generate_tutoring_prompt(student_learning_file, user_request)

    # [수정] 생성된 프롬프트를 AI API에 전달하여 실제 답변을 생성
    ai_response = get_ai_response(final_prompt)
    
    # [수정] AI의 최종 답변을 터미널에 출력
    print("\n" + "="*50)
    print("           AI 과외 선생님의 맞춤형 학습 계획")
    print("="*50 + "\n")
    print(ai_response)
