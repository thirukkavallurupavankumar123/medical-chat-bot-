from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY   

embeddings=download_hugging_face_embeddings()

index_name="medical-bot"

docsearch=PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatmodel=ChatGroq(model="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

question_answering_chain=create_stuff_documents_chain(chatmodel,prompt)
rag_chain=create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answering_chain)





@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("response: ",response["answer"])

    return str(response["answer"])




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
