import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 

load_dotenv()

def update_knowledge():
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    persist_directory = "./chroma_langchain_db"
    collection_name = "example_collection"
    
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        
        try:
            shutil.rmtree(persist_directory)
            print("Folder deleted successfully.")
        except Exception as e:
            print(f"Error while deleting folder: {e}")
            
        print("Existing vector store deleted.")
        
    print("Creating new vector store...")
    loader = TextLoader("store_knowledge.txt", encoding="utf-8")
    docs = loader.load()

    # 4. ตัดแบ่งข้อมูล
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150
    )
    all_splits = text_splitter.split_documents(docs)

    # 5. บันทึกลง Vector Database ใหม่
    print(f"--- Creating new database with {len(all_splits)} chunks ---")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name="example_collection",
        persist_directory=persist_directory
    )
    
    print("Update Completed.")
    
if __name__ == "__main__":
    update_knowledge()