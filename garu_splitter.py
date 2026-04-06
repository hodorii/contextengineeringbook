import os
import garu_core
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document


class GaruTextSplitter(TextSplitter):
    """Garu 기반 한국어 텍스트 스플리터"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_path: str = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "garu", "models"
        )

        codebook_path = os.path.join(model_path, "codebook.gmdl")
        cnn_path = os.path.join(model_path, "cnn2.bin")

        with open(codebook_path, "rb") as f:
            model_data = f.read()
        with open(cnn_path, "rb") as f:
            cnn_data = f.read()

        self._garu = garu_core.Garu(model_data, cnn_data)

    def split_text(self, text: str) -> list:
        tokens = self._garu.tokenize(text)
        return tokens

    def create_documents(self, texts: list, metadatas: list = None) -> list[Document]:
        raise NotImplementedError("Use split_text instead")
