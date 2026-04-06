import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from llm_proxy import LLMProxy, EmbeddingProxy
import garu_core


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


def get_garu_splitter():
    """Garu 토크나이저를 초기화합니다."""
    model_path = os.environ.get("GARU_MODEL_PATH", "/Users/hodorii/dev/garu/models")

    with open(os.path.join(model_path, "codebook.gmdl"), "rb") as f:
        model_data = f.read()
    with open(os.path.join(model_path, "cnn2.bin"), "rb") as f:
        cnn_data = f.read()

    return garu_core.Garu(model_data, cnn_data)


def split_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_garu: bool = None,
) -> list:
    """문서를 청크로 분할합니다."""
    if use_garu is None:
        use_garu = os.environ.get("USE_GARU", "true").lower() == "true"

    if use_garu:
        garu = get_garu_splitter()

        all_chunks = []
        for doc in documents:
            tokens = garu.tokenize(doc.page_content)
            text_with_separators = " 。 ".join(tokens)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[" 。 ", "\n\n", "\n", " "],
                length_function=len,
            )
            chunks = text_splitter.split_text(text_with_separators)

            for chunk in chunks:
                chunk = chunk.replace(" 。 ", " ")
                all_chunks.append(
                    Document(page_content=chunk.strip(), metadata=doc.metadata)
                )

        return all_chunks
    else:
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


def create_rag_chain(vector_store: Chroma):
    """RAG 체인을 생성합니다."""
    proxy = LLMProxy(provider="google")
    llm = proxy.llm

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

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

컨тек스트:
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
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def query_rag(question: str, vector_store: Chroma = None, directory: str = ".") -> str:
    """RAG를 사용하여 질문에 답변합니다."""
    if vector_store is None:
        documents = load_documents_from_directory(directory)
        splits = split_documents(documents)
        vector_store = create_vector_store(splits)

    rag_chain = create_rag_chain(vector_store)
    return rag_chain.invoke({"question": question})


def build_rag_pipeline(directory: str = ".", persist_directory: str = "./chroma_db"):
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
    rag_chain = create_rag_chain(vector_store)

    return rag_chain, vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    question = "이 프로젝트의 주요 기능은 무엇인가요?"

    result = query_rag(question)
    print(f"\nQ: {question}\nA: {result}")
