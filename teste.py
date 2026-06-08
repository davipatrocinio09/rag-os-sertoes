from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os
# CHAVE DA OPENAIsk
os.environ["opent"] ="minha chave "

# CARREGAR PDF
loader = PyPDFLoader("data/os-sertoes.pdf")
documents = loader.load()

print("os sertoes.pdf")

# CHUNKS 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents[:20])

print(f"Chunks criados: {len(chunks)}")

# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings carregados")

# FAISS
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("Banco vetorial criado")

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("LLM carregada")

# FUNÇÃO RAG
def ask_question(query):

    docs = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Use o contexto abaixo para responder.

CONTEXTO:
{context}

PERGUNTA:
{query}

Responda em português.
"""

    response = llm.invoke(prompt)

    return response.content

# TESTE
resposta = ask_question(
    "Quem foi Antônio Conselheiro?"
)

print("\nRESPOSTA:\n")
print(resposta)

print("Começo da chave:", os.environ["OPENAI_API_KEY"][:10])

