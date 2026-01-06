"""
임베딩 모델 차원 진단 스크립트
현재 설정에서 실제로 어떤 차원이 사용되는지 확인합니다.
"""
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("임베딩 모델 차원 진단")
print("=" * 60)

# FixedDimensionEmbeddings 클래스 정의
class FixedDimensionEmbeddings(Embeddings):
    def __init__(self, target_dimension: int = 1536):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.target_dimension = target_dimension
        self.model = "text-embedding-3-small"
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.target_dimension
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.target_dimension
        )
        return response.data[0].embedding

# 테스트 1: FixedDimensionEmbeddings
print("\n[테스트 1] FixedDimensionEmbeddings(target_dimension=1536)")
try:
    emb1 = FixedDimensionEmbeddings(target_dimension=1536)
    vec1 = emb1.embed_query("test")
    print(f"✓ 차원: {len(vec1)}")
except Exception as e:
    print(f"✗ 오류: {e}")

# 테스트 2: 기본 OpenAIEmbeddings
print("\n[테스트 2] OpenAIEmbeddings()")
try:
    emb2 = OpenAIEmbeddings()
    vec2 = emb2.embed_query("test")
    print(f"✓ 차원: {len(vec2)}")
    print(f"  모델: {emb2.model}")
except Exception as e:
    print(f"✗ 오류: {e}")

# 테스트 3: dimensions 파라미터 테스트
print("\n[테스트 3] OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)")
try:
    emb3 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vec3 = emb3.embed_query("test")
    print(f"✓ 차원: {len(vec3)}")
    print(f"  모델: {emb3.model}")
    if len(vec3) != 1536:
        print(f"  ⚠️  dimensions 파라미터가 무시되었습니다!")
except Exception as e:
    print(f"✗ 오류: {e}")

# 기존 DB 확인
print("\n" + "=" * 60)
print("기존 ChromaDB 확인")
print("=" * 60)

for db_name in ["experience_db", "experience_db_v2"]:
    if os.path.exists(db_name):
        try:
            from langchain_community.vectorstores import Chroma
            
            vs = Chroma(
                persist_directory=db_name,
                embedding_function=FixedDimensionEmbeddings(1536),
                collection_name="code_refactoring_experiences"
            )
            
            count = vs._collection.count()
            print(f"\n[{db_name}]")
            print(f"  문서 개수: {count}")
            
            if count > 0:
                result = vs._collection.get(limit=1, include=['embeddings'])
                if result['embeddings'] and len(result['embeddings']) > 0:
                    existing_dim = len(result['embeddings'][0])
                    print(f"  저장된 임베딩 차원: {existing_dim}")
                    if existing_dim != 1536:
                        print(f"  ⚠️  차원 불일치! (기대: 1536, 실제: {existing_dim})")
                else:
                    print("  임베딩 정보 없음")
        except Exception as e:
            print(f"\n[{db_name}] 읽기 오류: {e}")
    else:
        print(f"\n[{db_name}] 폴더가 존재하지 않습니다.")

print("\n" + "=" * 60)
print("해결 방법")
print("=" * 60)
print("""
상황별 해결 방법:

1. FixedDimensionEmbeddings가 1536차원을 생성하는 경우:
   → 기존 DB 삭제: rm -rf experience_db experience_db_v2
   
2. FixedDimensionEmbeddings가 384차원을 생성하는 경우:
   → OpenAI API 키 확인 또는 네트워크 문제
   
3. dimensions 파라미터가 무시되는 경우:
   → LangChain 버전 문제, FixedDimensionEmbeddings 사용 권장
""")
