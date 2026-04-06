import os
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# .env 파일 로드를 위한 라이브러리 임포트 (Best Practice)
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 1. 의뢰인 상담 기록
consultation_log = """
문서 유형: 의뢰인 상담 기록
출처: 상담 기록 #2025-08-20

- 의뢰인: 김민준
- 구매 제품: Quantum GFX 9000 그래픽카드
- 구매일: 2025-03-15
- 구매 금액: 2,000,000원
- 고장 발생일: 2025-08-18 (구매 후 약 5개월)
- 증상: 게임 중 화면 깨짐 현상 발생 후 PC 전원 꺼짐. 이후 부팅 불가.
- 판매자 측 주장: "제품 보증 기간은 1개월이다. 기간이 경과하여 환불/교환 불가."
- 의뢰인 요구사항: 제품 전액 환불 또는 동일 새 제품으로 교환.
"""

# 2. 관련 법령
civil_act_580 = """
문서 유형: 법령
출처: 민법 제580조 (매도인의 하자담보책임)

① 매매의 목적물에 하자가 있는 때에는 제575조 제1항의 규정을 준용한다. 그러나 매수인이 하자있는 것을 알았거나 과실로 인하여 이를 알지 못한 때에는 그러하지 아니하다.
"""

civil_act_582 = """
문서 유형: 법령
출처: 민법 제582조 (전조의 권리행사기간)

전2조에 의한 권리는 매수인이 그 사실을 안 날로부터 6월내에 행사하여야 한다.
"""

# 3. 변호사가 직접 선별한 핵심 판례
key_precedent = """
문서 유형: 판례
출처: 대법원 2021다12345 판결 요지

약관에 기재된 짧은 보증 기간이, 민법상 하자담보책임에 관한 매수인의 권리를 부당하게 배제하거나 제한하는 경우, 해당 약관 조항은 '약관의 규제에 관한 법률'에 위반되어 무효로 볼 수 있다. 특히, 고가의 전자제품에서 통상적인 사용 환경 하에 단기간 내에 발생하는 중대한 기능적 하자는, 매수인이 구매 시점에서는 쉽게 알 수 없는 '숨은 하자'에 해당할 가능성이 높다. 이러한 경우, 판매자의 1개월 보증 기간 주장만으로 매수인의 법적 권리를 박탈할 수 없다. 매수인은 하자를 안 날로부터 6개월 내에 정당한 권리를 행사할 수 있다.
"""

# 모든 문서를 하나의 리스트로 결합
case_documents = [consultation_log, civil_act_580, civil_act_582, key_precedent]


# --- 원본 코드 시작 ---

# --- RAG Indexing with Metadata ---

# 1. 문서를 LangChain의 Document 객체로 변환하면서 '출처' 메타데이터 추가
langchain_docs = []
for doc_text in case_documents:
    # 각 문서의 첫 줄에서 '출처:' 정보를 파싱하여 메타데이터로 저장
    lines = doc_text.strip().splitlines()
    if len(lines) > 2 and "출처: " in lines[1]:
        source = lines[1].replace("출처: ", "")
        page_content = "\n".join(lines[2:]) # 실제 내용
        langchain_docs.append(Document(page_content=page_content, metadata={"source": source}))
    else:
        # 양식에 맞지 않는 문서는 일반 문서로 처리
        langchain_docs.append(Document(page_content=doc_text.strip()))

# 2. 텍스트를 청크로 나누기 (여기서는 문서가 짧아 생략 가능하지만, 긴 문서를 위해 포함)
# CLAUDE/GEMINI: 이 텍스트 분할 로직(LangChain)은 모델에 상관없이 동일하게 사용됩니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(langchain_docs)

print(f"사건 파일이 {len(split_docs)}개의 정보 조각으로 인덱싱 준비 완료.")

# 3. 청크들을 임베딩하여 벡터 DB에 저장
embeddings = OpenAIEmbeddings()
# CLAUDE: Anthropic 임베딩 모델로 변경합니다.
# CLAUDE: embeddings = AnthropicEmbeddings() 
# GEMINI: Google 임베딩 모델로 변경합니다. (모델명 지정 필수)
# GEMINI: embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# CLAUDE/GEMINI: FAISS는 어떤 임베딩 모델이든 상관없이 작동하므로 이 코드는 변경할 필요가 없습니다.
vector_db = FAISS.from_documents(split_docs, embeddings)

print("'출처 정보'가 포함된 사건 파일 벡터 DB 생성 완료.")


# 1. '출처 명시'를 강제하는 프롬프트 템플릿 정의
PROMPT_TEMPLATE = """
당신은 대한민국 법률에 정통한 유능한 AI 법률 비서입니다.
아래에 제공된 "[배경 자료]"만을 근거로 하여 "[질문]"에 대해 답변하세요.
절대 당신의 사전 지식을 사용하지 마세요.

답변은 법률 의견서 초안 형식으로, 서론, 본론, 결론으로 나누어 논리적으로 작성해야 합니다.
가장 중요한 규칙은, 모든 핵심적인 주장이나 사실 관계 언급 끝에는 반드시 그것이 근거한 배경 자료의 출처를 "[출처: ...]" 형식으로 명시해야 한다는 것입니다.

만약 배경 자료에 질문에 대한 답변의 근거가 전혀 없다면, "제공된 자료만으로는 판단할 수 없음"이라고만 답변하세요.

[배경 자료]
{summaries}

[질문]
{question}

[답변]
"""

# 2. 프롬프트 템플릿 객체 생성
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["summaries", "question"])

# 3. '출처'를 함께 반환하는 RetrievalQAWithSourcesChain 체인 생성
# 이 체인은 RAG 검색 결과와 그 출처(메타데이터)를 함께 처리하는 데 특화되어 있습니다.
legal_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
    # [수정] llm: OpenAI(BaseLLM) -> ChatOpenAI(ChatModel)로 변경
    llm=ChatOpenAI(model=”gpt-5.1”, reasoning={ "effort": "none" }, max_tokens=2000),
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)
# 2. llm 객체를 ChatAnthropic으로 교체
# CLAUDE: legal_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
# CLAUDE: llm=ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0,
max_tokens=2000), 
# CLAUDE:     chain_type="stuff",
# CLAUDE:     retriever=vector_db.as_retriever(),
# CLAUDE:     chain_type_kwargs={"prompt": PROMPT}
# CLAUDE: )
# 2. llm 객체를 ChatGoogleGenerativeAI로 교체
# GEMINI: legal_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
# GEMINI: llm=ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0, max_
output_tokens=2000),
# GEMINI:     chain_type="stuff",
# GEMINI:     retriever=vector_db.as_retriever(),
# GEMINI:     chain_type_kwargs={"prompt": PROMPT}
# GEMINI: )

# 변호사의 질문
query = "판매자의 1개월 보증 기간 정책을 근거로 한 환불 거부가 타당한지 검토하고, 우리 측이 주장할 수 있는 법적 근거를 정리하여 법률 의견서 초안을 작성해 줘."

# AI 법률 비서 실행
# (최신 LangChain에서는 .run() 대신 .invoke()를 권장합니다)
result = legal_assistant_chain.invoke({"question": query})

print("\n" + "="*50)
print("                  AI가 생성한 법률 의견서 초안")
print("="*50 + "\n")
print(result['answer'])
print("\n" + "="*50)
