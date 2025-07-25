from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os


from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def rag_application(user_question: str):
    # Lendo o arquivo .txt
    loader = TextLoader("cesupa.txt", encoding="utf-8")
    documentos = loader.load()

    # Dividindo o texto em partes menores (chunks)
    split = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = split.split_documents(documentos)

    # Gerando embeddings para cada parte
    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
    )

    # Armazena os embeddings no ChromaDB
    vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./advanced_chroma_db",
            collection_name="cesupa-knowledge"
    )

    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
    )


    prompt_template = hub.pull("langchain-ai/retrieval-qa-chat", include_model=True)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Criando a stuff chain (combine_docs_chain)
    stuff_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )

    # Criando uma retrieval chain (RAG)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_chain
    )

    # Para consultar:
    resposta = retrieval_chain.invoke({"input": user_question})

    return resposta["answer"]


if __name__ == "__main__":
    # Exemplo de uso
    resposta = rag_application("O que voce sabe sobre o cesupa?")
    print(resposta)  # Deve imprimir a resposta gerada pelo modelo para a pergunta
