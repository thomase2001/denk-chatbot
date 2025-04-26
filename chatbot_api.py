# chatbot_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1️⃣ Load your OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2️⃣ Load your scraped content
loader = TextLoader("denk_live.txt", encoding="utf-8")
documents = loader.load()

# 3️⃣ Split documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 4️⃣ Create vector database
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# 5️⃣ Create RetrievalQA chatbot
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 6️⃣ FastAPI app setup
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_bot(query: QueryRequest):
    result = qa.run(query.question)
    return {"answer": result}
