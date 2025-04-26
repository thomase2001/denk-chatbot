from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1️⃣ Load your OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2️⃣ Load your scraped content
loader = TextLoader("denk_live.txt", encoding="utf-8")
documents = loader.load()

# 3️⃣ Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 4️⃣ Create vector database
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# 5️⃣ Create a Retrieval-based QA system
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 6️⃣ FastAPI app
app = FastAPI()

# Pydantic model for incoming requests
class QueryRequest(BaseModel):
    question: str

# 7️⃣ Main chatbot route
@app.post("/ask")
def ask_bot(query: QueryRequest):
    # Try to answer from your document knowledge
    result = qa.run(query.question)

    # Simple way to check if the result is empty or generic
    if not result or "I don't know" in result or "Sorry" in result:
        # If not good, ask ChatGPT directly
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        openai_result = llm.invoke(query.question)
        return {"answer": openai_result.content}

    return {"answer": result}

