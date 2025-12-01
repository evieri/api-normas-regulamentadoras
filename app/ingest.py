import os
import time
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(BASE_DIR, "base")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

def ingest_data():
    print("--- INICIANDO PIPELINE DE INGESTÃO DE DADOS ---")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERRO: GOOGLE_API_KEY não encontrada no .env")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print(f"Lendo PDFs da pasta: {PDF_FOLDER}")
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Pasta '{PDF_FOLDER}' criada. Adicione seus PDFs lá e rode novamente.")
        return

    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()
    
    if not docs:
        print("Nenhum documento encontrado.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    db = None
    print("Iniciando vetorização (Batch processing)...")

    for i, chunk in enumerate(chunks):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if db is None:
                    db = FAISS.from_documents([chunk], embeddings)
                else:
                    db.add_documents([chunk])
                
                sys.stdout.write(f"\rProcessando chunk {i+1}/{len(chunks)}")
                sys.stdout.flush()
                
                time.sleep(2) 
                break
            except Exception as e:
                retry_count += 1
                wait_time = retry_count * 10
                print(f"\nErro no chunk {i+1}: {e}. Aguardando {wait_time}s...")
                time.sleep(wait_time)
        
        if retry_count == max_retries:
            print(f"\nFalha crítica no chunk {i+1}. Salvando progresso parcial.")
            if db: db.save_local(INDEX_PATH)
            return

    if db:
        db.save_local(INDEX_PATH)
        print(f"\n\nSUCESSO: Índice FAISS salvo em '{INDEX_PATH}'")

if __name__ == "__main__":
    ingest_data()