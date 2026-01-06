from langchain.agents import tool

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

# 사용할 도구들을 리스트로 묶기
developer_tools = [read_file, write_file]