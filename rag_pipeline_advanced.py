import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_proxy import LLMProxy, EmbeddingProxy

from rank_bm25 import BM25Okapi
import numpy as np


def load_documents_from_directory(directory: str = ".") -> list:
    """디렉터리에서 모든 .py 파일을 로드합니다."""
    documents = []
    directory_path = Path(directory)

    for py_file in directory_path.glob("*.py"):
        if py_file.name.startswith("."):
            continue
        try:
            loader = TextLoader(str(py_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata = {"source": py_file.name}
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {py_file}: {e}")

    return documents


def split_documents(
    documents: list, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """문서를 청크로 분할합니다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vector_store(documents: list, persist_directory: str = None) -> Chroma:
    """벡터 저장소를 생성합니다."""
    embedding_proxy = EmbeddingProxy(provider="huggingface")

    if persist_directory is None:
        persist_directory = tempfile.mkdtemp()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_proxy.embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def load_or_create_vector_store(
    documents: list = None,
    persist_directory: str = "./chroma_db",
    force_recreate: bool = False,
) -> Chroma:
    """기존 벡터 저장소가 있으면 로드, 없으면 생성합니다."""
    embedding_proxy = EmbeddingProxy(provider="huggingface")

    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Loading existing vector store from {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_proxy.embeddings,
        )

    if documents is None:
        raise ValueError("documents must be provided when creating new vector store")

    print(f"Creating new vector store at {persist_directory}")
    return create_vector_store(documents, persist_directory)


class BM25Retriever:
    """BM25 기반 검색기 (LangChain 호환)"""

    def __init__(self, documents: list, k: int = 3):
        self.documents = documents
        self.k = k
        self._build_index()

    def _build_index(self):
        """BM25 인덱스 구축"""
        texts = [doc.page_content for doc in self.documents]
        # 토큰화 (공백 기준으로 단순 토큰화)
        self.tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_relevant_documents(self, query: str) -> list:
        """검색 수행"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 인덱스 가져오기
        top_indices = np.argsort(scores)[::-1][: self.k]

        return [self.documents[i] for i in top_indices]

    def invoke(self, query: str) -> list:
        """LangChain 호환 메서드"""
        return self.get_relevant_documents(query)


class EnsembleRetriever:
    """BM25 + Vector (앙상블) 검색기"""

    def __init__(self, vector_retriever, bm25_retriever, weights: tuple = (0.5, 0.5)):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights  # (vector_weight, bm25_weight)

    def invoke(self, query: str) -> list:
        """앙상블 검색 수행"""
        # 벡터 검색 결과
        vector_docs = self.vector_retriever.invoke(query)

        # BM25 검색 결과
        bm25_docs = self.bm25_retriever.invoke(query)

        # 결과 병합 (중복 제거しながら)
        combined = []
        seen_contents = set()

        for doc in vector_docs:
            if doc.page_content not in seen_contents:
                combined.append(doc)
                seen_contents.add(doc.page_content)

        for doc in bm25_docs:
            if doc.page_content not in seen_contents:
                combined.append(doc)
                seen_contents.add(doc.page_content)

        return combined[:3]


def create_rag_chain(
    vector_store: Chroma,
    documents: list = None,
    use_bm25: bool = True,
    use_ensemble: bool = True,
):
    """RAG 체인을 생성합니다."""
    proxy = LLMProxy(provider="google")
    llm = proxy.llm

    # 벡터 검색기
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # BM25 검색기 (활성화 시)
    if use_bm25 and documents:
        bm25_retriever = BM25Retriever(documents, k=3)

        if use_ensemble:
            # 앙상블 사용 (벡터 50% + BM25 50%)
            retriever = EnsembleRetriever(
                vector_retriever, bm25_retriever, weights=(0.5, 0.5)
            )
        else:
            retriever = bm25_retriever
    else:
        retriever = vector_retriever

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 주어진 컨텍스트(문서)를 바탕으로 질문에 답변하는 어시스턴트입니다.
다음 규칙을 따라주세요:
1. 제공된 컨텍스트에만 기반하여 답변하세요
2. 컨텍스트에 정보가 없으면 "컨텍스트에 정보가 없습니다"라고 말씀하세요
3. 코드 예시를 포함하여 설명하면 좋습니다.
4. 답변은 한국어로 작성해주세요.

컨텍스트:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def query_rag(
    question: str,
    vector_store: Chroma = None,
    documents: list = None,
    directory: str = ".",
    use_bm25: bool = True,
    use_ensemble: bool = True,
) -> str:
    """RAG를 사용하여 질문에 답변합니다."""
    if vector_store is None:
        documents = load_documents_from_directory(directory)
        splits = split_documents(documents)
        vector_store = create_vector_store(splits)
        documents = splits  # BM25용

    rag_chain = create_rag_chain(vector_store, documents, use_bm25, use_ensemble)
    return rag_chain.invoke({"question": question})


def build_rag_pipeline(
    directory: str = ".",
    persist_directory: str = "./chroma_db",
    use_bm25: bool = True,
    use_ensemble: bool = True,
):
    """전체 RAG 파이프라인을 빌드합니다."""
    print("Loading documents...")
    documents = load_documents_from_directory(directory)
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    splits = split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Creating vector store...")
    vector_store = load_or_create_vector_store(splits, persist_directory)

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(vector_store, splits, use_bm25, use_ensemble)

    return rag_chain, vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    question = "이 프로젝트의 주요 기능은 무엇인가요?"

    # 다양한 검색 방식 테스트
    print("=" * 50)
    print("1. Vector Search Only")
    print("=" * 50)
    result = query_rag(question, use_bm25=False, use_ensemble=False)
    print(f"\nQ: {question}\nA: {result}\n")

    print("=" * 50)
    print("2. BM25 Only")
    print("=" * 50)
    result = query_rag(question, use_bm25=True, use_ensemble=False)
    print(f"\nQ: {question}\nA: {result}\n")

    print("=" * 50)
    print("3. Ensemble (Vector + BM25)")
    print("=" * 50)
    result = query_rag(question, use_bm25=True, use_ensemble=True)
    print(f"\nQ: {question}\nA: {result}")
