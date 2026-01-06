from dotenv import load_dotenv
import os
import openai

load_dotenv()
# open ai api key 확인
#print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")
# open ai 버전 확인
print(f"[open ai version]\n{openai.__version__}")
