# API de Análise Documental de Normas Regulamentadoras com RAG

Este projeto consiste em uma **API RESTful** desenvolvida para análise inteligente de documentos técnicos (ex: Normas Regulamentadoras, Editais). O sistema utiliza **IA Generativa** e arquitetura **RAG (Retrieval-Augmented Generation)** para fornecer respostas contextualizadas com citação de fontes.

Projeto desenvolvido como prova de conceito (PoC) para competências de Engenharia de IA.

## 🚀 Tecnologias Utilizadas

* **Backend:** Python 3.12, FastAPI.
* **IA Generativa:** Google Gemini 1.5 Flash (via `langchain-google-genai`).
* **RAG & Orquestração:** LangChain.
* **Banco Vetorial:** FAISS (Facebook AI Similarity Search) com indexação local.
* **Engenharia de Dados:** Chunking recursivo e tratamento de Rate Limits.

## ⚙️ Funcionalidades

* **Ingestão de PDFs:** Processamento de documentos brutos, limpeza e divisão em chunks semânticos.
* **Busca Semântica:** Recuperação de trechos relevantes baseada em similaridade vetorial (Embeddings).
* **API de Perguntas:** Endpoint `POST /api/v1/ask` que recebe uma pergunta e retorna a resposta gerada pela LLM + as fontes consultadas.
* **Source Tracking:** O sistema indica exatamente quais trechos do documento foram usados para gerar a resposta, reduzindo alucinações.

## 🛠️ Instalação e Execução

### 1. Configuração do Ambiente
```bash
# Clone o repositório
git clone [https://github.com/evieri/api-normas-regulamentadoras](https://github.com/evieri/api-normas-regulamentadoras)
cd api-normas regulamentadoras

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt