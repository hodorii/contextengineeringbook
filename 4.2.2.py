import os
from llm_proxy import LLMProxy, EmbeddingProxy

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 가상 제품 매뉴얼 텍스트 데이터
product_manual_text = """
# 에어쿨러 V2 사용 설명서

## 1. 기본 사용법
에어쿨러 V2는 전원 버튼을 눌러 켤 수 있습니다. 바람 세기는 1단계(수면풍), 2단계(자연풍), 3단계(터보풍)로 조절 가능합니다. 리모컨의 '바람 세기' 버튼을 누를 때마다 단계가 순환됩니다.

## 2. AI 절전 모드 상세 설명
본 제품의 핵심 기능인 'AI 절전 모드'는 실내 온도와 습도를 5분마다 자동으로 감지하여 최적의 냉방 수준을 유지함으로써 전기 요금을 최대 40%까지 절약할 수 있습니다.
- 활성화 방법: 리모컨의 'AI 모드' 버튼을 3초간 길게 누릅니다. LED 창에 'AI' 문구가 표시되면 활성화된 것입니다.
- 작동 원리:
  - 실내 온도가 26도 이상일 경우: 자동으로 터보풍으로 작동하여 신속하게 온도를 낮춥니다.
  - 실내 온도가 23도에서 25도 사이일 경우: 자연풍으로 전환하여 쾌적한 상태를 유지합니다.
  - 실내 온도가 22도 이하로 내려갈 경우: 수면풍으로 자동 전환되거나, 전원이 자동으로 꺼져 불필요한 에너지 소비를 방지합니다.
- 주의 사항: 'AI 절전 모드' 사용 중에는 수동으로 바람 세기 조절이 불가능합니다. 모드를 해제하려면 'AI 모드' 버튼을 다시 짧게 한 번 누르면 됩니다.

## 3. 문제 해결 (FAQ)
Q1: 전원이 켜지지 않아요.
A1: 전원 코드가 제대로 연결되어 있는지 확인해 주세요. 멀티탭을 사용 중이라면, 멀티탭의 전원이 켜져 있는지도 확인이 필요합니다.

Q2: 리모컨이 작동하지 않아요.
A2: 배터리가 방전되었을 수 있습니다. AAA 사이즈 배터리 2개를 새것으로 교체해 보세요.
"""

# --- 원본 코드 시작 ---

# --- RAG Indexing: 지식 창고 만들기 ---

# 1. 텍스트를 의미 있는 단위(Chunk)로 쪼갭니다.
# 헤더("#", "##")를 기준으로 문단을 나누면 의미 단위로 잘 나눌 수 있습니다.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
split_docs = text_splitter.split_text(product_manual_text)

print(f"제품 매뉴얼이 {len(split_docs)}개의 정보 조각(청크)으로 나뉘었습니다.")

# 2. 쪼갠 청크들을 임베딩하여 벡터 DB에 저장합니다.
llm_proxy = LLMProxy(provider="google")
embedding_proxy = EmbeddingProxy(provider="huggingface")
vector_db = FAISS.from_documents(split_docs, embedding_proxy.embeddings)

print("지식 창고(벡터 DB)가 성공적으로 생성되었습니다.")

# 3. RAG 체인을 생성합니다. 이 체인이 '제품 전문가'의 역할을 수행합니다.
product_expert_chain = RetrievalQA.from_chain_type(
    llm=llm_proxy.llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    return_source_documents=True,  # 답변의 근거가 된 원본 문서를 함께 반환하도록 설정
)
# CLAUDE/GEMINI: LangChain의 RetrievalQA 체인 구성 코드는 모델에 상관없이 동일합니다.


# --- 4. RAG 체인 실행 테스트 ---
def format_sources(source_docs):
    """검색된 출처 문서들을 보기 좋게 포맷팅합니다."""
    sources = []
    for i, doc in enumerate(source_docs, 1):
        header = doc.metadata.get("Header 1") or doc.metadata.get(
            "Header 2", "알 수 없는 섹션"
        )
        content = doc.page_content[:100].replace("\n", " ")
        sources.append(f"  [{i}] {header}: {content}...")
    return "\n".join(sources)


if __name__ == "__main__":
    print("\nRAG 시스템이 준비되었습니다. 질문을 입력하세요.\n")

    query1 = "AI 절전 모드는 어떻게 켜나요?"
    answer1 = product_expert_chain.invoke(query1)

    print(f"❓ 질문: {query1}")
    print(f"✅ 답변: {answer1['result']}")
    print(f"📚 참조 출처:")
    print(format_sources(answer1["source_documents"]))
    print()

    query2 = "리모컨이 안 먹혀요"
    answer2 = product_expert_chain.invoke(query2)

    print(f"❓ 질문: {query2}")
    print(f"✅ 답변: {answer2['result']}")
    print(f"📚 참조 출처:")
    print(format_sources(answer2["source_documents"]))
