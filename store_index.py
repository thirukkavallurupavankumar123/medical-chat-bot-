from dotenv import load_dotenv
from src.helper import load_pdf_files, filtered_tp_minimal_docs, text_splitter, download_hugging_face_embeddings
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data=load_pdf_files(data="data/")
fiter_data=filtered_tp_minimal_docs(extracted_data)
text_chunks=text_splitter(fiter_data)

embeddings=download_hugging_face_embeddings()
pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-bot"

if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index=pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(documents=text_chunks, embedding=embeddings, index_name=index_name)
