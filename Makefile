.PHONY: help query build rebuild clean test install install-garu

PYTHON = .venv/bin/python
GARU_MODEL_PATH ?= /Users/hodorii/dev/garu/models
USE_GARU ?= true

help:
	@echo "Available targets:"
	@echo "  query        - Run RAG query (usage: make query QUESTION='your question')"
	@echo "  build        - Build vector store from current directory"
	@echo "  rebuild      - Rebuild vector store from scratch"
	@echo "  clean        - Remove vector store"
	@echo "  test         - Run test query"
	@echo "  install      - Install Python dependencies"
	@echo "  install-garu - Build and install garu-core Python module"
	@echo ""
	@echo "Options:"
	@echo "  USE_GARU=true|false   - Use Garu tokenizer (default: true)"
	@echo "  GARU_MODEL_PATH=...    - Path to Garu model files (default: /Users/hodorii/dev/garu/models)"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-garu:
	@echo "Building garu-core..."
	cd /Users/hodorii/dev/garu/crates/garu-core && \
		PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 $(PYTHON) -m maturin build --release && \
		$(PYTHON) -m pip install target/wheels/garu_core-*.whl
	@echo "garu-core installed successfully"

build:
	$(PYTHON) -c "from rag_pipeline import build_rag_pipeline; build_rag_pipeline()"
	@echo "Vector store built successfully"

rebuild:
	$(PYTHON) -c "import shutil, os; [shutil.rmtree('./chroma_db') for f in ['./chroma_db'] if os.path.exists(f)]; from rag_pipeline import build_rag_pipeline; build_rag_pipeline()"
	@echo "Vector store rebuilt successfully"

query:
ifndef QUESTION
	$(error QUESTION is undefined. Usage: make query QUESTION='your question')
endif
	$(PYTHON) -c "\
import os; \
os.environ['GARU_MODEL_PATH'] = '$(GARU_MODEL_PATH)'; \
os.environ['USE_GARU'] = '$(USE_GARU)'; \
from rag_pipeline import load_or_create_vector_store, create_rag_chain, load_documents_from_directory, split_documents; \
from dotenv import load_dotenv; \
load_dotenv(); \
documents = load_documents_from_directory('.'); \
use_garu = '$(USE_GARU)'.lower() == 'true'; \
splits = split_documents(documents, use_garu=use_garu); \
vs = load_or_create_vector_store(splits); \
chain = create_rag_chain(vs); \
result = chain.invoke({'question': '$(QUESTION)'}); \
print(result)"

test:
	make query QUESTION="이 프로젝트의 주요 기능은 무엇인가요?"

clean:
	$(PYTHON) -c "import shutil, os; [shutil.rmtree('./chroma_db') for f in ['./chroma_db'] if os.path.exists(f) and print('chroma_db removed')] or print('No chroma_db to remove')"
