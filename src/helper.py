from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List
from langchain_core.documents import Document
import os

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
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

