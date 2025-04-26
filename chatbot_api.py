# chatbot_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware

# 1️⃣ FastAPI app
app = FastAPI()

# 2️⃣ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or replace with your Wix domain for more security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Load your OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 4️⃣ Load your scraped content
loader = TextLoader("denk_live.txt", encoding="utf-8")
documents = loader.load()

# 5️⃣ Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 6️⃣ Create vector database
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# 7️⃣ Create a Retrieval-based QA system
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 8️⃣ Pydantic model for incoming requests
class QueryRequest(BaseModel):
    question: str

# 9️⃣ Main chatbot route
@app.post("/ask")
def ask_bot(query: QueryRequest):
    # First, try to answer based on the document
    result = qa.run(query.question)

    # If no good answer, fallback to direct ChatGPT
    if not result or "I don't know" in result or "Sorry" in result:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        openai_result = llm.invoke(query.question)
        return {"answer": openai_result.content}

    return {"answer": result}

# 🔟 (Optional) Root route to prevent 404 if people open "/"
@app.get("/")
def read_root():
    return {"message": "DENK.bot läuft 🚀"}

