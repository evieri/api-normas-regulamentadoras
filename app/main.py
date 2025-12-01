import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
app = FastAPI(
    title="DocuMind API",
    description="API de RAG para consulta documental inteligente",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

class QuestionRequest(BaseModel):
    query: str

class Source(BaseModel):
    content: str
    page: int

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]

vector_store = None

@app.on_event("startup")
async def load_vector_store():
    """Carrega o banco vetorial na memória ao iniciar a API"""
    global vector_store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(INDEX_PATH):
        try:
            vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("INFO: Banco vetorial FAISS carregado com sucesso.")
        except Exception as e:
            print(f"ERRO: Falha ao carregar banco vetorial: {e}")
    else:
        print("WARN: Índice não encontrado. Execute 'python app/ingest.py' primeiro.")

@app.post("/api/v1/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Banco vetorial não disponível. O serviço está indexando ou falhou.")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
        system_prompt = (
            "Você é um assistente especialista. Use os contextos abaixo para responder à pergunta. "
            "Se não souber, diga que a informação não consta nos documentos. "
            "Mantenha a resposta técnica e direta.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": request.query})

        sources = []
        for doc in response["context"]:
            sources.append(Source(
                content=doc.page_content[:200] + "...",
                page=doc.metadata.get("page", 0) + 1
            ))

        return AnswerResponse(
            answer=response["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "online", "model": "Gemini 1.5 Flash"}