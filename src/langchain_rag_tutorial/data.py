from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader

from langchain_rag_tutorial.util import STATE_OF_THE_UNION_TXT, VECTOR_DIR


def ingest_data() -> None:
    print("Loading data …")
    docs = UnstructuredLoader(STATE_OF_THE_UNION_TXT).load()

    print("Splitting text …")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    print("Creating vectorstore …")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(docs, embeddings)

    vectordb.save_local(str(VECTOR_DIR))
    print(f"Vector store saved to {VECTOR_DIR.resolve()}")
