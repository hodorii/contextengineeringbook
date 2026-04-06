import os
import json
from dotenv import load_dotenv

# --- 1. LangChain 및 LLM 라이브러리 임포트 ---
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- 2. 환경 설정: API 키 ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")

# --- 3. 핵심 컴포넌트 정의: LLM, Embeddings ---
llm = ChatOpenAI(model=”gpt-5.1”, reasoning={ "effort": "low" })
embeddings = OpenAIEmbeddings()

# CLAUDE: llm = ChatAnthropic(model=”claude-sonnet-4-5-20250929”, temperature=0.1)
# CLAUDE: embeddings = AnthropicEmbeddings()
# GEMINI: llm = ChatGoogleGenerativeAI(model=”gemini-3.0”, temperature=0.1)
# GEMINI: embeddings = GoogleGenerativeAIEmbeddings(model=”models/embedding-001”)


def get_ai_response(prompt: str) -> str:
    """최종 프롬프트를 LLM에 전달하여 응답을 생성하는 함수"""
    try:
        # LangChain의 .invoke() 메서드를 사용하여 챗 모델 호출
        response = llm.invoke([{"role": "user", "content": prompt}])
        if isinstance(response.content, list):
            # 리스트 일 경우
            # return "".join([str(block) for block in response.content]).strip()
            # 딕셔너리 구조일 경우 (예상)
            return "".join([
                block.get('text', '') if isinstance(block, dict) else str(block) 
                for block in response.content
            ]).strip()
        else:
            # 문자열인 경우 기존대로 처리
            return response.content.strip()
    except Exception as e:
        return f"AI 응답 생성 중 오류 발생: {e}"


# --- 4. 컴포넌트 A: 동적 페르소나 (사용자 제공 코드) ---
persona_library = {
    "problem_solver": """
    당신은 고객의 문제를 해결하는 데 집중하는 침착하고 공감 능력 높은 '문제 해결 전문가'입니다.
    - 목표: 고객의 불만을 해소하고, 신뢰를 회복하는 것.
    - 말투: 차분하고, 정중하며, 고객의 입장을 이해한다는 점을 표현하세요. ("많이 답답하셨겠어요.", "불편을 드려 정말 죄송합니다.")
    """,
    "tech_supporter": """
    당신은 제품의 기술적인 질문에 답변하는 명료하고 정확한 '기술 지원 전문가'입니다.
    - 목표: 고객이 제품의 기능을 완벽하게 이해하고 사용할 수 있도록 돕는 것.
    - 말투: 전문적이고, 자신감 있으며, 군더더기 없이 명확하게 설명하세요.
    """,
    "brand_ambassador": """
    당신은 우리 브랜드에 애정을 가진 고객과 소통하는 밝고 친근한 '브랜드 앰버서더'입니다.
    - 목표: 고객의 긍정적인 경험을 강화하고, 브랜드 충성도를 높이는 것.
    - 말투: 밝고, 활기차며, 감사를 적극적으로 표현하세요.
    """,
    "default": "당신은 친절하고 상냥한 일반 상담원입니다. 고객의 질문에 명확하고 간결하게 답변하세요."
}

def classify_intent(message):
    """사용자 메시지에서 키워드를 기반으로 의도를 분류하는 함수"""
    message = message.lower()
    problem_keywords = ["환불", "고장", "안돼요", "불만", "작동 안 함", "문제"]
    if any(keyword in message for keyword in problem_keywords):
        return "problem_solver"
    tech_keywords = ["어떻게", "방법", "기능", "설정", "매뉴얼", "사용법"]
    if any(keyword in message for keyword in tech_keywords):
        return "tech_supporter"
    positive_keywords = ["좋아요", "감사합니다", "최고", "만족", "잘 쓰고 있어요"]
    if any(keyword in message for keyword in positive_keywords):
        return "brand_ambassador"
    return "default"

# --- 5. 컴포넌트 B: 고객 기억 (CRM 연동) ---
# (가상 CRM DB 및 연동 함수 구현)
mock_crm_db = {
    "C-1001": {
        "name": "김민준", 
        "level": "VIP", 
        "recent_order_id": "ORD-5678",
        "past_inquiries": ["배송 지연 문의 (2025-10-01)"]
    },
    "C-2002": {
        "name": "이서연", 
        "level": "GOLD", 
        "recent_order_id": "ORD-5679",
        "past_inquiries": []
    },
}

def get_customer_history(customer_id: str) -> dict:
    """[구현] 가상 CRM DB에서 고객 데이터를 조회합니다."""
    return mock_crm_db.get(customer_id, {})

def format_history_as_context(customer_data: dict) -> str:
    """[구현] 조회된 고객 데이터를 LLM이 이해하기 쉬운 컨텍스트로 변환합니다."""
    if not customer_data:
        return "신규 고객 또는 비회원 고객입니다."
    
    context = f"""
    - 고객명: {customer_data.get('name', 'N/A')}
    - 고객 등급: {customer_data.get('level', 'N/A')}
    - 최근 주문번호: {customer_data.get('recent_order_id', 'N/A')}
    - 과거 문의 내역: {', '.join(customer_data.get('past_inquiries', [])) if customer_data.get('past_inquiries') else '없음'}
    """
    return context.strip()

# --- 6. 컴포넌트 C: 전문 지식 (RAG) ---
# (가상 제품 매뉴얼 텍스트 - 환불 정책 추가)
product_manual_text = """
# 에어쿨러 V2 사용 설명서
## 1. 기본 사용법
에어쿨러 V2는 전원 버튼을 눌러 켤 수 있습니다. 바람 세기는 1단계(수면풍), 2단계(자연풍), 3단계(터보풍)로 조절 가능합니다.

## 2. AI 절전 모드 상세 설명
본 제품의 핵심 기능인 'AI 절전 모드'는 실내 온도를 자동으로 감지하여 전기 요금을 최대 40%까지 절약할 수 있습니다. 
- 활성화 방법: 리모컨의 'AI 모드' 버튼을 3초간 길게 누릅니다.

## 3. 문제 해결 (FAQ)
Q1: 전원이 켜지지 않아요.
A1: 전원 코드가 제대로 연결되어 있는지 확인해 주세요.

# 알파 전자 고객센터 운영 규정
## 1. 환불 규정
- VIP 고객은 구매 후 14일, 일반 고객은 7일 이내에 환불 가능합니다.
- 단, 제품의 포장이 개봉되었거나 사용 흔적이 있는 경우 환불이 불가능합니다.
"""

def setup_rag_pipeline(manual_text: str, embeddings) -> RetrievalQA:
    """[구현] 텍스트를 받아 RAG 체인을 생성하고 반환합니다."""
    print("\n--- RAG 지식 베이스(product_expert_chain) 생성 중 ---")
    
    # 1. 텍스트 분할
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_docs = text_splitter.split_text(manual_text)
    
    # 2. 벡터 DB 생성
    vector_db = FAISS.from_documents(split_docs, embeddings)
    
    # 3. RAG 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        return_source_documents=True
    )
    print("--- RAG 지식 베이스 생성 완료 ---")
    return qa_chain

# --- 7. '완전체' AI 챗봇 함수 (사용자 제공 코드 수정) ---

def ultimate_chatbot(customer_id, message, product_expert_chain):
    """
    CRM, RAG, 동적 페르소나를 모두 통합하여
    최종 프롬프트를 생성하고, LLM을 호출하여 답변을 반환합니다.
    """
    
    # 1. 고객 기억 (CRM 연동)
    customer_data = get_customer_history(customer_id)
    customer_context = format_history_as_context(customer_data)
    
    # 2. 전문 지식 (RAG)
    # product_expert_chain은 RAG 체인 객체
    rag_result = product_expert_chain.invoke(message)
    knowledge_context = rag_result['result']
    # source_documents = rag_result['source_documents'] # 필요시 출처 표시
    
    # 3. 공감 능력 (동적 페르소나)
    intent = classify_intent(message)
    selected_persona = persona_library[intent]
    
    # 4. 모든 컨텍스트를 결합하여 최종 프롬프트 생성
    final_prompt = f"""
    {selected_persona}

    ---
    ### 고객 정보 브리핑
    {customer_context}
    ---
    ### 관련 매뉴얼 정보
    {knowledge_context}
    ---
    ### 고객 현재 문의
    {message}
    """
    
    print("\n--- [Debug] 최종 통합 프롬프트 ---")
    print(final_prompt)
    
    # 5. [수정] 이 최종 프롬프트를 LLM에 전달하여 답변 생성
    final_response = get_ai_response(final_prompt)
    return final_response


# --- 8. 시나리오별 시뮬레이션 ---
if __name__ == "__main__":
    
    # 1. RAG 파이프라인(제품 전문가)을 미리 준비
    product_expert_chain = setup_rag_pipeline(product_manual_text, embeddings)
    
    # 2. 시나리오 실행: VIP 고객이 8일 후 환불 요청
    print("\n" + "="*50)
    print("===== 시나리오 1: VIP 환불 요청 =====")
    print("="*50)
    
    customer_id = "C-1001" # 김민준 VIP 고객
    message = "안녕하세요. 8일 전에 산 상품 환불하고 싶어요."
    
    response = ultimate_chatbot(customer_id, message, product_expert_chain)
    
    print("\n--- [최종 AI 답변] ---")
    print(response)

    # 3. 시나리오 실행: 일반 고객이 기술 문의
    print("\n" + "="*50)
    print("===== 시나리오 2: 일반 고객 기술 문의 =====")
    print("="*50)
    
    customer_id = "C-2002" # 이서연 GOLD 고객
    message = "AI 절전 모드 어떻게 켜요?"
    
    response = ultimate_chatbot(customer_id, message, product_expert_chain)
    
    print("\n--- [최종 AI 답변] ---")
    print(response)
