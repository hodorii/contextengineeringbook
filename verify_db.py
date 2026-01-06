import chromadb

def inspect_knowledge_base():
    """
    ChromaDB에 저장된 지식 베이스의 현재 상태를 확인하는 스크립트.
    """
    print("--- 지식 베이스 상태 검증 시작 ---")
    
    try:
        # 에이전트가 사용한 동일한 DB 폴더에 연결
        db_client = chromadb.PersistentClient(path="medical_knowledge_db")
        collection = db_client.get_collection(name="papers")
        
        # 1. 지식의 양 확인
        total_docs = collection.count()
        print(f"\n[검증 1] 현재 지식 베이스에 저장된 총 문서 수: {total_docs}개")
        
        if total_docs == 0:
            print(">> 아직 학습된 지식이 없습니다.")
            return

        # 2. 저장된 지식 내용 확인
        print("\n[검증 2] 저장된 전체 문서 내용:")
        
        # get() 함수로 DB의 모든 내용을 불러옴
        all_data = collection.get() 
        
        for i in range(total_docs):
            doc_id = all_data['ids'][i]
            document = all_data['documents'][i]
            metadata = all_data['metadatas'][i]
            
            print(f"\n--- 문서 ID: {doc_id} ---")
            print(f"  내용: {document}")
            # print(f"  메타데이터: {metadata}") # 필요시 주석 해제하여 상세 메타데이터 확인
            
    except Exception as e:
        print(f"검증 중 오류 발생: {e}")
        print(">> 에이전트를 한 번 이상 실행하여 'medical_knowledge_db' 폴더가 생성되었는지 확인하세요.")

if __name__ == "__main__":
    inspect_knowledge_base()