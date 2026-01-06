import os
# OpenAI(BaseLLM) 대신 ChatOpenAI(ChatModel)를 임포트합니다.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# [수정] text_splitter가 'langchain-text-splitters' 패키지로 분리되었습니다.
# [수정] pip install langchain-text-splitters가 필요합니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # [수정] FAISS도 -community로 이동
from langchain.chains import RetrievalQA

# [추가] .env 파일 로드를 위한 라이브러리 임포트 (Best Practice)
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키를 환경 변수에 설정합니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 1. 지식 베이스가 될 원본 텍스트 데이터입니다.
knowledge_base_text = """
[알파 전자 고객센터 운영 규정]

제1조 (환불 규정)
- 모든 제품은 구매일로부터 14일 이내에 환불이 가능합니다.
- 단, 제품의 포장이 개봉되었거나 사용 흔적이 있는 경우 환불이 불가능합니다.
- 환불 접수는 공식 홈페이지의 '1:1 문의'를 통해서만 가능합니다.

제2조 (배송 정책)
- 평일 오후 3시 이전 결제 완료 건에 한해 당일 출고를 원칙으로 합니다.
- 기본 배송비는 3,000원이며, 5만원 이상 구매 시 무료 배송입니다.
- 제주 및 도서 산간 지역은 추가 배송비 4,000원이 부과됩니다.

제3조 (AS 정책)
- 제품 보증 기간은 구매일로부터 1년입니다.
- 소비자 과실로 인한 고장은 유상 수리 처리되며, 수리 비용은 모델별로 상이합니다.
- AS 접수는 전국 서비스센터 방문 또는 택배 접수를 통해 가능합니다.
"""

# --- Phase 1: 지식 창고 만들기 (Indexing) ---

# Step 1 & 2: 텍스트를 의미 있는 단위(Chunk)로 쪼갭니다.
# (이 부분은 Claude, Gemini에서도 동일하게 사용됩니다.)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(knowledge_base_text)

print(f"문서가 {len(chunks)}개의 청크로 나뉘었습니다.")

# Step 3 & 4: 쪼갠 청크들을 임베딩하여 벡터 DB에 저장합니다.
# OpenAI의 임베딩 모델을 사용합니다.
embeddings = OpenAIEmbeddings()
# CLAUDE: Anthropic 임베딩 모델로 변경합니다.
# CLAUDE: embeddings = AnthropicEmbeddings()
# GEMINI: Google 임베딩 모델로 변경합니다. (모델명 지정 필수)
# GEMINI: embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# FAISS를 사용하여 인메모리 벡터 DB를 생성합니다.
# (이 부분은 어떤 임베딩 모델을 쓰든 동일하게 작동합니다.)
vector_db = FAISS.from_texts(chunks, embeddings)

print("벡터 DB가 성공적으로 생성되었습니다.")

# --- Phase 2: 질문에 답변하기 (Retrieval & Generation) ---

# RAG 파이프라인을 생성합니다.
qa_chain = RetrievalQA.from_chain_type(
    # [수정] llm: OpenAI(BaseLLM) -> ChatOpenAI(ChatModel)로 변경
    llm=ChatOpenAI(model=”gpt-5.1”, reasoning={ "effort": "none" }),
    # CLAUDE: LLM 모델을 ChatAnthropic으로 변경합니다. (모델명 지정 필수)
    # CLAUDE: llm=ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0),
    # GEMINI: LLM 모델을 ChatGoogleGenerativeAI로 변경합니다. (모델명 지정 필수)
    # GEMINI: llm=ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0),

    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

print("\nRAG 시스템이 준비되었습니다. 질문을 입력하세요.\n")

# 사용자 질문
query1 = "포장을 뜯었는데 환불 되나요?"
query2 = "제주도 배송비는 얼마인가요?"

# 질문 실행 및 결과 확인
# (RAG 체인 실행 코드는 모든 모델에서 동일합니다.)
# (최신 LangChain에서는 .invoke()를 권장합니다.)
answer1 = qa_chain.invoke(query1)
answer2 = qa_chain.invoke(query2)

print(f"질문 1: {query1}")
print(f"답변 1: {answer1['result']}\n")

print(f"질문 2: {query2}")
print(f"답변 2: {answer2['result']}")
