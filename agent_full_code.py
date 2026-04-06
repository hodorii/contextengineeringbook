import os
import shutil
import json
import numpy as np
import chromadb # [핵심 수정] 'chromadb' 네이티브 클라이언트 임포트
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 로드 ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# [수정] 임베딩 모델 환경 변수 (OpenAIEmbeddings에서 model 파라미터가 변경됨)
# text-embedding-3-small은 1536차원이 기본입니다.
os.environ["EMBEDDING_MODEL_NAME"] = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
os.environ["EMBEDDING_MODEL_DIM"] = os.getenv("EMBEDDING_MODEL_DIM", "1536")

# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# --- 1. 실습 환경 자동 생성 ---
def setup_project_environment():
    # (원본과 동일)
    print("--- 1. 실습 환경 자동 생성 시작 ---")
    project_dir = "buggy_project"
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    os.makedirs(project_dir)
    with open(os.path.join(project_dir, "utils.py"), "w", encoding="utf-8") as f:
        f.write("""
def calculate_total(cart_items):
    \"\"\"장바구니 상품들의 총액을 계산합니다.\"\"\"
    total_price = 0
    for item in cart_items:
        total_price += item['price'] * item['quantity']
    return total_price
""")
    with open(os.path.join(project_dir, "main.py"), "w", encoding="utf-8") as f:
        f.write("""
from utils import calculate_total
cart = [
    {'name': '사과', 'price': 1500, 'quantity': 5},
    {'name': '바나나', 'price': 3000, 'quantity': '2'}
]
try:
    total = calculate_total(cart)
    print(f"총 합계: {total}")
except TypeError as e:
    print(f"오류가 발생했습니다: {e}")
""")
    print(f"'{project_dir}' 폴더와 내부에 `utils.py`, `main.py` 생성 완료.\n")


# --- 2. 에이전트의 '손과 발': 도구(Tools) 정의 ---
# (원본과 동일)
@tool
def read_file(file_path: str) -> str:
    """파일의 전체 내용을 읽을 때 사용합니다. 인자로는 파일 경로(문자열)를 받습니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 오류: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """파일에 새로운 내용을 쓰거나 수정할 때 사용합니다. 인자로는 파일 경로(문자열)와 새로운 내용(문자열)을 받습니다."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"'{file_path}' 파일이 성공적으로 수정되었습니다."
    except Exception as e:
        return f"파일 쓰기 오류: {e}"

developer_tools = [read_file, write_file]


# --- 3. [수정됨] 통합형 ExperienceDB ---
class ExperienceDB:
    """차원별 컬렉션 관리 + 메타 검증 + 선택적 재임베딩 통합형"""

    def __init__(self, persist_dir="experience_db", embeddings=None, base_collection="experiences"):
        self.persist_dir = persist_dir
        self.embeddings = embeddings # LangChain 래퍼 객체
        self.base_collection = base_collection
        self.registry_path = os.path.join(persist_dir, "registry.json")

        os.makedirs(persist_dir, exist_ok=True)
        self.current_dim = getattr(self.embeddings, "dimensions", 1536) # 3-small 기본값
        self.current_model = getattr(self.embeddings, "model", "unknown-model")

        self.registry = self._load_registry()
        self.collection_name = self._resolve_collection()
        
        self.client = chromadb.PersistentClient(path=self.persist_dir) # 네이티브 클라이언트
        
        self.vectorstore = self._load_or_create_collection() # LangChain 래퍼
        self._register_collection()
        print(f"--- 3. ChromaDB 경험 데이터베이스 '{self.collection_name}' 준비 완료 ---")


    def _load_registry(self):
        # (원본과 동일)
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_registry(self):
        # (원본과 동일)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _register_collection(self):
        # (원본과 동일)
        if not any(c["name"] == self.collection_name for c in self.registry):
            self.registry.append({
                "name": self.collection_name,
                "dim": self.current_dim,
                "model": self.current_model,
            })
            self._save_registry()

    def _resolve_collection(self):
        # (원본과 동일)
        for c in self.registry:
            if c["model"] == self.current_model and c["dim"] == self.current_dim:
                return c["name"]
        safe_model_name = self.current_model.replace('-', '_').replace('.', '_')
        return f"{self.base_collection}_dim{self.current_dim}_model_{safe_model_name}"

    def _load_or_create_collection(self):
        # [핵심 수정] 
        collection_exists_in_registry = any(c["name"] == self.collection_name for c in self.registry)
        
        if collection_exists_in_registry:
            try:
                print(f"[정보] 기존 컬렉션 '{self.collection_name}' 로드를 시도합니다...")
                # 네이티브 클라이언트로 메타데이터를 먼저 확인
                native_collection = self.client.get_collection(
                    name=self.collection_name
                )
                meta = native_collection.metadata
                
                if meta is None:
                    print("[정보] 로드된 컬렉션에 메타데이터가 없습니다. 새로 설정합니다.")
                    native_collection.modify(metadata={"model": self.current_model, "dimension": self.current_dim})
                else:
                    stored_dim = meta.get("dimension")
                    if stored_dim and stored_dim != self.current_dim:
                        print(f"[경고] 차원 불일치! 기존 {stored_dim}, 현재 {self.current_dim}. 새 컬렉션을 생성합니다.")
                        self.collection_name = f"{self.base_collection}_dim{self.current_dim}_model_{self.current_model.replace('-', '_')}_new"
                        return self._create_new_collection()

                print(f"[성공] 기존 컬렉션 '{self.collection_name}' 로드 완료.")
                
            except Exception as e:
                print(f"[경고] 레지스트리에 있으나 로드 실패, 새로 생성: {e}")
                self.collection_name = f"{self.base_collection}_dim{self.current_dim}_model_{self.current_model.replace('-', '_')}_new"
                return self._create_new_collection()
        else:
            print(f"[정보] 새 컬렉션 '{self.collection_name}'을(를) 생성합니다.")
            return self._create_new_collection()

        # 성공적으로 로드/확인된 컬렉션을 LangChain 래퍼로 감싸서 반환
        return Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )


    def _create_new_collection(self):
        """새 Chroma 컬렉션을 생성하고 메타데이터를 설정하는 내부 함수"""
        # 네이티브 클라이언트를 사용하여 메타데이터와 함께 생성
        # [수정] LangChain 임베딩 래퍼는 네이티브 함수가 아님.
        #       LangChain 래퍼를 통해 생성하는 것이 더 안전함.
        print(f"[정보] _create_new_collection 호출: '{self.collection_name}'")
        new_store = Chroma.from_texts(
            texts=["[SYSTEM_INIT]"], # 컬렉션 생성을 위한 초기 더미 데이터
            embedding=self.embeddings,
            metadatas=[{"init": True}],
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )
        # 네이티브 컬렉션에 메타데이터 설정
        new_store._collection.modify(metadata={"model": self.current_model, "dimension": self.current_dim})
        new_store.persist()
        return new_store


    def add_experience(self, text, metadata=None):
        self.vectorstore.add_texts([text], metadatas=[metadata or {}])
        self.vectorstore.persist() 

    def query_experience(self, query, k=1):
        try:
            if self.vectorstore._collection.count() == 0:
                print("[정보] 경험 DB가 비어있어 검색을 건너뜁니다.")
                return ""
            
            results = self.vectorstore.similarity_search_with_score(
                query, 
                k=k, 
                filter={"feedback": "y"} 
            )
            
            if not results:
                print("[정보] 유사한 '성공' 사례를 찾지 못했습니다.")
                return ""
            
            doc, score = results[0]
            content = doc.page_content 
            meta = doc.metadata
            
            print(f"✅ 유사한 성공 경험(유사도: {score:.4f})을 찾아 컨텍스트를 강화했습니다.")
            return f"\n### 참고할 성공 사례 (과거 작업 로그)\n{content}\n"
        
        except Exception as e:
            print(f"[오류] 경험 검색 실패: {e}")
            return ""

    def list_collections(self):
        return [f"{c['name']} ({c['model']} · dim={c['dim']})" for c in self.registry]
    
    def reembed_from(self, old_collection_name):
         pass # (구현 생략)


# --- 4. 코드 리팩터링 에이전트 ---
class CodeRefactoringAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
        self.embedding_model = os.environ["EMBEDDING_MODEL_NAME"]
        self.embedding_dim = int(os.environ["EMBEDDING_MODEL_DIM"])
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model, 
            dimensions=self.embedding_dim
        )
        print(f"임베딩 모델: {self.embedding_model} ({self.embedding_dim}차원)")
        
        # CLAUDE: self.llm = ChatAnthropic(...)
        # CLAUDE: self.embeddings = AnthropicEmbeddings()
        # GEMINI: self.llm = ChatGoogleGenerativeAI(...)
        # GEMINI: self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.experience_db = ExperienceDB(persist_dir="experience_db", embeddings=self.embeddings)
        print("--- 4. AI 주니어 개발자 에이전트 초기화 완료 ---\n")

    def _create_context_bundle(self, error_log, source_file_path):
        source_code = read_file.func(source_file_path)
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

    def run(self, bug_report):
        # (원본과 동일)
        print("--- 5. 버그 리포트 접수 및 작업 시작 ---")
        initial_context = self._create_context_bundle(
            bug_report["error_log"], bug_report["file_path"]
        )

        reinforced_context = self.experience_db.query_experience(initial_context)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 숙련된 파이썬 개발자 에이전트입니다. 주어진 컨텍스트와 성공 사례를 바탕으로 버그를 수정하세요.
ReadFile, WriteFile 도구를 사용하여 문제를 해결하세요.

{reinforced_context}
{initial_context}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, developer_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=developer_tools,
            memory=memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        try:
            result = agent_executor.invoke({"input": bug_report["goal"]})
            print(f"\n에이전트 결과:\n{result['output']}")
        except Exception as e:
            print(f"\n에이전트 실행 오류: {e}")
            return

        while True:
            feedback = input("\n이 수정이 올바른가요? (y/n): ").lower()
            if feedback in ['y', 'n']:
                # [핵심 수정]
                # 'trace' (에이전트의 작업 로그)를 'text' (문서 본문)로 저장합니다.
                # 'initial_context' (문제)를 'query' (검색 키)로 사용합니다.
                # 'feedback'을 'metadata'로 저장하여 필터링에 사용합니다.
                self.experience_db.add_experience(
                    text=str(memory.buffer), # [수정] trace를 본문으로 저장
                    metadata={"feedback": feedback.lower(), "original_query": initial_context}
                )
                break
            print("y 또는 n으로 입력해주세요.")

# --- 메인 실행 ---
if __name__ == "__main__": 
    setup_project_environment()
    agent_manager = CodeRefactoringAgent()

    # 수정: 에러 로그 가독성 개선
    bug_report_1 = {
        "error_log": """
Traceback (most recent call last):
  File "buggy_project/main.py", line 11, in <module>
    total = calculate_total(cart)
  File "buggy_project/utils.py", line 8, in calculate_total
    total_price += item['price'] * item['quantity']
TypeError: can't multiply sequence by non-int of type 'str'
""",
        "file_path": "buggy_project/utils.py",
        "goal": "이 버그 리포트를 분석하고, utils.py 파일의 TypeError를 수정해줘."
    }

    agent_manager.run(bug_report_1)