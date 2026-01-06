import os
from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

# [추가] .env 파일 로드를 위한 라이브러리 임포트 (Best Practice)
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 1. 답변 생성에 사용할 언어 모델(LLM)을 정의합니다.
llm = ChatOpenAI(model=”gpt-5.1”, reasoning={ "effort": "none" })

# CLAUDE: LLM 모델을 ChatAnthropic으로 변경합니다. (모델명 지정 필수)
# CLAUDE: llm = ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0)

# GEMINI: LLM 모델을 ChatGoogleGenerativeAI로 변경합니다. (모델명 지정 필수)
# GEMINI: llm = ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0)


# 2. 대화 요약 메모리를 생성합니다.
# llm을 메모리 객체에 넘겨주면, 이 llm이 대화를 요약하는 역할을 합니다.
# CLAUDE/GEMINI: 이 코드는 LangChain의 추상화 덕분에 변경할 필요가 없습니다.
# CLAUDE/GEMINI: 어떤 LLM 객체(ChatOpenAI, ChatAnthropic 등)를 전달하든 동일하게 작동합니다.
memory = ConversationSummaryMemory(llm=llm)

# 3. 메모리를 사용하는 대화 체인(Chain)을 생성합니다.
# verbose=True로 설정하면, 체인이 어떻게 작동하는지 내부 과정을 볼 수 있습니다.
# CLAUDE/GEMINI: 이 코드도 LangChain의 추상화 덕분에 변경할 필요가 없습니다.
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# --- 대화 시작 ---

# 첫 번째 대화
# 최신 LangChain은 .invoke()를 사용하며, ConversationChain은 딕셔너리 입력을 받습니다.
print("--- 1번째 대화 ---")
conversation_with_summary.invoke({"input": "안녕? 내 이름은 이서준이야. 나는 AI 기술을 활용한 챗봇 개발에 관심이 많아."})

# 두 번째 대화
print("\n--- 2번째 대화 ---")
conversation_with_summary.invoke({"input": "특히 RAG 기술에 대해 더 깊이 알고 싶어."})

# 세 번째 대화
# AI는 이전 대화 내용을 요약해서 기억하고 있으므로, '그 기술'이 RAG를 의미하는지 압니다.
print("\n--- 3번째 대화 ---")
response = conversation_with_summary.invoke({"input": "그 기술을 우리 회사 고객 지원팀에 어떻게 적용할 수 있을까?"})

print("\n--- 최종 응답 ---")
print(response['response'])
