from fastapi import FastAPI
import getpass
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent


load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API_KEY")

# model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model_ollama = ChatOllama(model="granite4.1:8b")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = None
agent_executor = None

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """ใช้สำหรับค้นหาข้อมูลเกี่ยวกับร้าน The Brew Logix (เดอะ บริว โลจิกซ์)"""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = (
        "คุณคือผู้ช่วย AI อัจฉริยะประจำร้านกาแฟ 'The Brew Logix' (เดอะ บริว โลจิกซ์) "
        "คุณมีหน้าที่ตอบคำถามลูกค้าโดยใช้เครื่องมือ retrieve_context เพื่อค้นหาข้อมูลที่ถูกต้อง "
        "คำแนะนำในการตอบ:\n"
        "1. ทุกครั้งที่ลูกค้าถามเกี่ยวกับร้าน ให้เรียกใช้เครื่องมือเพื่อหาคำตอบเสมอ\n"
        "2. ถ้าข้อมูลที่ดึงมาไม่มีคำตอบที่ลูกค้าต้องการ ให้ตอบสุภาพว่า 'ขออภัยครับ ผมไม่มีข้อมูลส่วนนี้' หรือ 'ลองติดต่อเบอร์ร้านโดยตรง'\n"
        "3. ตอบคำถามด้วยความเป็นมิตรและใช้ภาษาไทยที่เป็นกันเองแต่สุภาพ\n"
        "4. สนใจเฉพาะข้อมูลที่เป็นข้อเท็จจริงจากเครื่องมือเท่านั้น ไม่ต้องแต่งเนื้อหาเอง"
    )

agent_executor = create_agent(model_ollama, tools, system_prompt=prompt)

class ChatRequest(BaseModel):
    query: str

app = FastAPI()

def load_system():
    
    global vector_store, agent_executor
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    
    tools = [retrieve_context]
    # เรียกใช้ Agent ที่สร้างไว้แล้ว
    agent_executor = create_agent(model_ollama, tools, system_prompt=prompt)
    
@app.on_event("startup")
def startup_event():
    load_system()

@app.post("/chat")
def chat(request: ChatRequest):
    result = agent_executor.invoke({"messages": [("user", request.query)]})

    return {
        "query": request.query,
        "answer": result["messages"][-1].content
    }
    
@app.post("/update-knowledge")
def update_knowledge():
    # เรียกใช้ฟังก์ชัน update_knowledge จากไฟล์ ingest.py
    from ingest import update_knowledge
    
    try:
        update_knowledge()
        load_system()
        return {"message": "Knowledge updated and system reloaded successfully."}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    