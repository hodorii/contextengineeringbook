import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# [수정] 최신 LangChain 경로로 수정
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# [추가] .env 파일 로드를 위한 라이브러리 임포트 (Best Practice)
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: Gos.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 1. 익명화된 환자 EMR 데이터
anonymized_emr = """
문서 유형: 익명화된 EMR 요약
출처: 환자 ID #A-1234-5678

- 진단명: 비소세포폐암 (NSCLC), 4기
- 나이/성별: 62세/남성
- 주요 유전자 변이: EGFR Exon 19 결실 (EGFR Exon 19 Deletion)
- 현재 상태: 1세대 EGFR TKI(표적치료제) '게피티닙' 치료 14개월 후 내성 발현 및 암 진행 확인.
- 추가 검사 결과: 내성의 주요 원인인 T790M 변이는 '음성(Negative)'임. 대신, 드문 유형인 'Exon 19 C-나선 변이'가 관찰됨.
- 특이사항: 심장 기능 저하 소견 있음.
"""

# 2. 의사가 직접 선별한 최신 연구 논문 (가상)
research_paper_1 = """
문서 유형: 연구 논문
출처: NEJM 2025; 392:41-51 (타그리소 연구)

- 제목: T790M 변이 양성 NSCLC 환자에서 3세대 TKI '오시머티닙(타그리소)'의 효능
- 요약: 1, 2세대 EGFR TKI 치료에 실패하고 T790M 내성 변이가 생긴 비소세포폐암 환자에서, 오시머티닙은 높은 반응률(ORR 71%)과 긴 무진행 생존기간(PFS 10.1개월)을 보였다.
"""

research_paper_2 = """
문서 유형: 연구 논문
출처: Lancet Oncology 2025; 26:112-122 (렉라자 연구)

- 제목: EGFR 변이 NSCLC의 1차 치료로서 '레이저티닙(렉라자)'의 효과
- 요약: EGFR 변이가 있는 비소세포폐암 환자의 1차 치료에서 레이저티닙은 기존 치료제 대비 우수한 뇌전이 억제 효과를 보였다. T790M 내성 환자에 대한 데이터는 제한적이다.
"""

research_paper_3 = """
문서 유형: 연구 논문
출처: JCO 2025; 43:150-159 (신약 OncoVax-7 연구)

- 제목: EGFR TKI 내성 및 희귀 변이 NSCLC 환자에서 차세대 면역항암제 'OncoVax-7'의 2상 임상시험 결과
- 요약: 표준 EGFR TKI 치료에 내성이 생겼으며, 특히 T790M 음성이면서 Exon 19 C-나선 변이와 같은 희귀 변이를 가진 환자군(n=30)에서 'OncoVax-7'은 45%의 객관적 반응률(ORR)을 보였다. 주요 부작용으로는 면역 관련 심근염이 보고되어, 심장 기능 모니터링이 필수적이다. 현재 3상 임상시험 참여자를 모집 중이다.
"""

# 모든 문서를 하나의 리스트로 결합
medical_documents = [anonymized_emr, research_paper_1, research_paper_2, research_paper_3]


# 1. 문서를 메타데이터와 함께 Document 객체로 변환
langchain_docs = []
for doc_text in medical_documents:
    # (가정) 소스코드의 파싱 로직은 medical_documents의 형식에 맞게 작성됨
    lines = doc_text.strip().splitlines()
    if len(lines) > 2 and "출처: " in lines[1]:
        source = lines[1].replace("출처: ", "")
        page_content = "\n".join(lines[2:])
        langchain_docs.append(Document(page_content=page_content, metadata={"source": source}))
    else:
        langchain_docs.append(Document(page_content=doc_text.strip()))


# 2. 텍스트를 청크로 나누고 벡터 DB에 저장
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(langchain_docs)

print(f"'익명화된 EMR'과 '최신 논문'이 {len(split_docs)}개의 정보 조각으로 인덱싱 준비 완료.")

# 3. 청크들을 임베딩하여 벡터 DB에 저장
embeddings = OpenAIEmbeddings()
# CLAUDE: Anthropic 임베딩 모델로 변경합니다.
# CLAUDE: embeddings = AnthropicEmbeddings() 
# GEMINI: Google 임베딩 모델로 변경합니다. (모델명 지정 필수)
# GEMINI: embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_db = FAISS.from_documents(split_docs, embeddings)

print("'출처 정보'가 포함된 의료 지식 벡터 DB 생성 완료.")


# 의료 연구 보조원의 역할을 정의하고 안전장치를 포함한 프롬프트 템플릿
PROMPT_TEMPLATE = """
당신은 의료 전문가(의사)를 보조하는 AI 의료 연구 보조원입니다. 당신의 임무는 제공된 [환자 데이터]와 [의학 문헌]을 종합하고 분석하여, 의사가 임상적 결정을 내리는 데 도움이 될 수 있는 객관적인 정보를 요약하는 것입니다.

**매우 중요한 규칙:**
1.  절대 직접적인 진단, 예후 예측, 또는 치료법 추천을 해서는 안 됩니다. 당신은 의사가 아니며, 모든 답변은 의사의 전문적 판단을 돕기 위한 정보 제공에 국한되어야 합니다.
2.  모든 정보는 반드시 제공된 [환자 데이터]와 [의학 문헌]에 근거해야 합니다. 당신의 사전 지식을 절대 사용하지 마세요.
3.  모든 핵심 내용 끝에는 반드시 근거 자료의 출처를 "[출처: ...]" 형식으로 명시해야 합니다.
4.  답변은 의사가 보고서 형식으로 볼 수 있도록 명확하고 구조화된 형태로 작성하세요.

[환자 데이터 및 의학 문헌]
{summaries}

[의사의 질문]
{question}

[AI 연구 보조원의 요약 보고서]
"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["summaries", "question"])

# OpenAI RAG 체인 생성
medical_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(model=”gpt-5.1”, reasoning={ "effort": "none" }, max_tokens=2000),
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}), # 관련성 높은 문서를 최대 5개까지 검색
    chain_type_kwargs={"prompt": PROMPT}
)

# CLAUDE RAG 체인 생성
# CLAUDE: medical_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
# CLAUDE: llm=ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0),
# CLAUDE:     chain_type="stuff",
# CLAUDE:     retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
# CLAUDE:     chain_type_kwargs={"prompt": PROMPT},
# CLAUDE:     input_key="question" # [핵심 수정]
# CLAUDE: )

# GEMINI RAG 체인 생성
# GEMINI: medical_assistant_chain = RetrievalQAWithSourcesChain.from_chain_type(
# GEMINI: llm=ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0),
# GEMINI:     chain_type="stuff",
# GEMINI:     retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
# GEMINI:     chain_type_kwargs={"prompt": PROMPT},
# GEMINI:     input_key="question" # [핵심 수정]
# GEMINI: )


# 의사의 질문
query = "이 환자의 EGFR T790M 음성, Exon 19 C-나선 변이 상태에서, 기존 EGFR TKI 내성 후 적용을 고려해볼 수 있는 치료 옵션에 대해 제공된 문헌들을 바탕으로 요약하고, 각 옵션의 근거와 주요 고려사항을 정리해 줘."

# AI 의료 연구 보조원 실행
# [수정] 체인의 입력 키를 'question'으로 통일했으므로 invoke()가 정상 작동합니다.
result = medical_assistant_chain.invoke({"question": query})

print("\n" + "="*50)
print("                  AI가 생성한 의료 정보 요약 보고서")
print("="*50 + "\n")
print(result['answer'])
