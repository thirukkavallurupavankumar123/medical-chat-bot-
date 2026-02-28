from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.documents import Document
from huggingface_hub import InferenceClient
import numpy as np
import os


class HFInferenceEmbeddings(Embeddings):
    """Custom embeddings using huggingface_hub InferenceClient."""
    
    def __init__(self, model: str, token: str):
        self.model = model
        self.client = InferenceClient(token=token)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            embedding = self.client.feature_extraction(text, model=self.model)
            results.append(np.array(embedding).flatten().tolist())
        return results
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.client.feature_extraction(text, model=self.model)
        return np.array(embedding).flatten().tolist()


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="**/*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
    
def filtered_tp_minimal_docs(docs: list[Document])-> list[Document]:
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content,metadata={"source":src}))
    return minimal_docs

def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = text_splitter.split_documents(minimal_docs)
    return split_docs

def download_hugging_face_embeddings():
    hf_token = os.environ.get("HF_TOKEN", "")
    embeddings = HFInferenceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token
    )
    return embeddings

