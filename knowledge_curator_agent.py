import os
import json
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# OpenAI: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings

# [수정] LangChain 0.1.0+ 버전에서는 에이전트 생성을 위한 핵심 모듈들이 분리되었습니다.
# Gemini와 같은 Tool Calling 모델을 위해 create_tool_calling_agent를 사용합니다.
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from pymed import PubMed
import chromadb
from apscheduler.schedulers.blocking import BlockingScheduler
# [수정] LangChain의 Chroma 래퍼를 임포트합니다.
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 ---
load_dotenv()
# Google API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

# --- 1. 도구(Tools) 정의 ---

# LLM, 임베딩, 벡터스토어를 전역 객체로 선언
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# OpenAI: llm = ChatOpenAI(model="gpt-4o", temperature=0)
# CLAUDE: llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# OpenAI: embeddings = OpenAIEmbeddings()
# CLAUDE: embeddings = AnthropicEmbeddings()

print("--- 1. 지식 베이스 초기화 중... ---")
vector_db = Chroma(
    persist_directory="medical_knowledge_db",
    embedding_function=embeddings,
    collection_name="papers"
)
print("지식 베이스(ChromaDB)가 성공적으로 준비되었습니다.\n")


@tool
def search_pubmed_for_new_papers(topic: str, max_results: int = 3) -> str:
    """[다중 인자 도구] PubMed에서 특정 주제(topic)에 대한 최신 논문을 검색합니다. 최근 7일 내의 논문만 검색합니다."""
    print(f"ACTION: PubMed 검색 (Topic: {topic})")
    try:
        pubmed = PubMed(tool="MyAIAgent", email="my_email@example.com")
        query = f'({topic}[Title/Abstract]) AND (("last 7 days"[Date - Publication]))'
        results = pubmed.query(query, max_results=max_results)
        paper_list = [{"title": r.title, "abstract": r.abstract, "doi": r.doi} for r in results if r.abstract]
        return json.dumps(paper_list) if paper_list else "새로운 논문을 찾지 못했습니다."
    except Exception as e:
        return f"PubMed 검색 중 오류 발생: {e}"

@tool
def summarize_and_refine_paper(abstract: str) -> str:
    """주어진 논문 초록(abstract)을 요약하고, DB에 저장할 형태로 정제합니다."""
    print("ACTION: 논문 요약 및 정제")
    prompt = f"다음 논문 초록을 이 연구의 핵심 결론을 담은 한 문장으로 요약해줘:\n\n{abstract}"
    
    try:
        response_obj = llm.invoke(prompt)
        summary = response_obj.content.strip()
        return summary
    except Exception as e:
        return f"요약 중 오류 발생: {e}"


@tool
def check_for_duplicates(summary: str, threshold: float = 0.95) -> bool:
    """새로운 요약문이 DB에 이미 존재하는 내용과 유사한지 확인합니다."""
    print("ACTION: 중복 데이터 검사")
    
    if vector_db._collection.count() == 0: 
        return False
    
    results = vector_db.similarity_search_with_score(summary, k=1)
    
    if results and results[0][1] > threshold:
        print("⚠중복된 내용 발견.")
        return True
    return False

@tool
def add_to_vector_db(summary: str, metadata: dict) -> str:
    """[다중 인자 도구] 정제된 요약문(summary)과 메타데이터(metadata)를 벡터 데이터베이스에 저장합니다."""
    print("ACTION: 벡터 DB에 신규 정보 저장")
    if check_for_duplicates.func(summary):
        return "중복된 내용이므로 저장하지 않았습니다."
    
    # DOI를 ID로 사용하거나, 없다면 요약문 자체를 ID로 사용 (더 안정적)
    doc_id = metadata.get("doi", summary[:50]) 
    
    vector_db.add_texts([summary], metadatas=[metadata], ids=[doc_id])
    vector_db.persist() # 변경 사항을 디스크에 즉시 저장
    
    return f"ID '{doc_id}'로 새로운 지식이 성공적으로 저장되었습니다."

curator_tools = [
    search_pubmed_for_new_papers,
    summarize_and_refine_paper,
    add_to_vector_db
]
# check_for_duplicates는 add_to_vector_db 내부에서만 사용

# --- 2. 에이전트 및 스케줄러 설정 ---

def create_curator_agent_executor():
    """ 지식 큐레이터 에이전트 실행기를 생성하는 함수"""
    
    # 1. 프롬프트 템플릿: ChatModel에 맞게 수정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 '지식 큐레이터' AI 에이전트입니다. 
        주어진 목표를 달성하기 위해 다음 도구들을 순서대로 사용해야 합니다: 
        1. `search_pubmed_for_new_papers`로 논문 리스트(JSON)를 검색합니다.
        2. 검색된 각 논문의 `abstract`에 대해 `summarize_and_refine_paper`를 호출하여 요약합니다.
        3. 각 요약문에 대해 `add_to_vector_db`를 호출하여, 원본 `title`과 `doi`를 'metadata'로 함께 저장합니다.
        최종 목표는 새로운 지식을 벡터 DB에 추가하는 것입니다."""),
        MessagesPlaceholder(variable_name="chat_history"), # 메모리용
        ("human", "{input}"), # 사용자 목표용
        MessagesPlaceholder(variable_name="agent_scratchpad") # 작업 공간용
    ])
    
    # 2. 에이전트 로직 생성
    # create_tool_calling_agent는 Gemini와 같은 모델에서 도구 호출을 효과적으로 지원합니다.
    agent = create_tool_calling_agent(llm, curator_tools, prompt)
    
    # 3. 메모리 생성
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 4. 에이전트 실행기(Executor) 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=curator_tools,
        memory=memory,
        verbose=True,
        max_iterations=15, # 여러 논문을 처리해야 하므로 반복 횟수 증가
        handle_parsing_errors=True
    )
    return agent_executor

def run_knowledge_curation_task(topic: str):
    """에이전트가 수행할 전체 작업을 정의한 함수 (Job)"""
    print(f"\n{'='*50}\n{topic} 주제에 대한 지식 큐레이션 작업을 시작합니다... ({time.ctime()})\n{'='*50}")
    
    # 에이전트 실행기 생성
    agent_executor = create_curator_agent_executor()
    goal = f"'{topic}'에 대한 최신 논문을 검색하여, 각 논문을 요약하고, 중복되지 않는 경우에만 지식 베이스에 추가하라."
    
    try:
        # [수정] .run(input=goal) 대신 .invoke({"input": goal}) 사용 (최신 방식)
        result = agent_executor.invoke({"input": goal})
        print(f"\n작업 완료: {result['output']}")
    except Exception as e:
        print(f"에이전트 실행 중 오류 발생: {e}")

# --- 3. 메인 실행 ---
if __name__ == "__main__":
    # 연구 주제 설정
    research_topic = "Alzheimer's disease biomarkers"
    
    # 1. 프로그램 시작 시 즉시 1회 실행
    run_knowledge_curation_task(research_topic)

    # 2. 스케줄러 설정 및 시작
    scheduler = BlockingScheduler()
    scheduler.add_job(run_knowledge_curation_task, 'interval', minutes=10, args=[research_topic])
    
    print("\n--- 스케줄러가 시작되었습니다. 10분마다 작업을 반복합니다. (Ctrl+C로 종료) ---")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n--- 스케줄러를 종료합니다. ---")