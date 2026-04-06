import os
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic

# [수정] LangChain 0.1.0+ 버전에서는 에이전트 생성을 위한 핵심 모듈들이 분리되었습니다.
# Gemini와 같은 Tool Calling 모델을 위해 create_tool_calling_agent를 사용합니다.
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from tools import developer_tools
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 로드 ---
load_dotenv()
# Google API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

# LLM (두뇌)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# OpenAI: llm = ChatOpenAI(model="gpt-4o", temperature=0)
# CLAUDE: llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Memory (기억)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # [수정] return_messages=True 추가

# --- 2. 버그 리포트 기반 Context Buffer 생성 ---
def create_context_bundle(error_log: str, source_file_path: str) -> str:
    """
    (이 함수는 LangChain이나 LLM API와 무관한 순수 Python 로직이므로
     어떤 모델을 사용하든 변경할 필요가 없습니다.)
    """
    source_code = ""
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception:
        source_code = f"Error: {source_file_path}를 읽을 수 없습니다."

    return f"""
### 초기 컨텍스트: 버그 리포트
당신의 임무는 아래 오류 로그와 관련 소스 코드를 분석하여 버그를 수정하는 것입니다.

#### 1. 오류 로그
```
{error_log}
```

#### 2. 관련 소스 코드 (`{source_file_path}`)
```python
{source_code}
```
""" 

# --- 3. 에이전트 초기화 (파이프라인 조립) ---
def create_refactoring_agent(initial_context: str):
    
    # 1. ChatModel에 맞는 프롬프트 템플릿(ChatPromptTemplate)을 정의합니다.
    # 'prefix'가 'system' 메시지가 됩니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""당신은 숙련된 파이썬 개발자입니다. 주어진 컨텍스트를 바탕으로 버그를 수정하세요. ReadFile, WriteFile 도구를 사용하여 문제를 해결하세요.
{initial_context}
"""),
        # 'memory_key'와 일치하는 "chat_history" 변수
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"), # 사용자의 입력을 받을 변수
        MessagesPlaceholder(variable_name="agent_scratchpad"), # 에이전트의 '생각'이 담길 변수
    ])
    
    # 2. LLM, 도구, 프롬프트를 사용하여 에이전트를 생성합니다.
    # create_tool_calling_agent는 Gemini 등 Tool Calling을 지원하는 모델에서 범용적으로 작동합니다.
    agent = create_tool_calling_agent(llm, developer_tools, prompt)
    
    # 3. 'AgentExecutor'를 사용하여 에이전트를 실행할 수 있는 객체를 만듭니다.
    # 이것이 'initialize_agent'를 대체하는 현대적인 방식입니다.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=developer_tools, 
        memory=memory, # 메모리 연결
        verbose=True
    )
    return agent_executor

# --- 4. 메인 실행 루프 (CLI + 피드백 루프) ---
if __name__ == "__main__":
    print("AI 주니어 개발자 에이전트가 준비되었습니다.")
    
    # 버그 리포트 정보
    error_log_from_report = """
Traceback (most recent call last):
  File "buggy_project/main.py", line 11, in <module>
    total = calculate_total(cart)
  File "buggy_project/utils.py", line 8, in calculate_total
    total_price += item['price'] * item['quantity']
TypeError: can't multiply sequence by non-int of type 'str'
"""
    source_file_to_fix = "buggy_project/utils.py"

    # Context Buffer 생성
    initial_context = create_context_bundle(error_log_from_report, source_file_to_fix)

    # 에이전트 생성
    agent = create_refactoring_agent(initial_context)

    # 초기 목표 전달
    user_goal = "이 버그 리포트를 분석하고, `utils.py` 파일의 TypeError를 수정해줘."

    # 에이전트 실행
    # [수정] .run(input=...) 대신, 딕셔너리를 받는 .invoke(...)를 사용합니다.
    agent_result = agent.invoke({"input": user_goal})
    # [수정] .run()은 문자열을 반환했지만, .invoke()는 딕셔너리를 반환합니다.
    #       최종 답변은 'output' 키에 들어있습니다.
    print(f"\n에이전트 작업 완료 보고:\n{agent_result['output']}")

    # 피드백 루프
    while True:
        feedback = input("\n이 수정이 올바른가요? (y/n): ").lower()
        if feedback in ['y', 'n']:
            # 8.6절에서 이 피드백을 '학습'에 활용하는 로직을 추가할 수 있습니다.
            print("피드백을 기록했습니다. 감사합니다.")
            break
        else:
            print("y 또는 n으로만 입력해주세요.")