from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_rag_tutorial.util import VECTOR_DIR

EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

vectordb = FAISS.load_local(
    folder_path=str(VECTOR_DIR),
    embeddings=EMBED_MODEL,
    allow_dangerous_deserialization=True,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using the following documents when relevant:\n{context}\n\nWhen you quote, append [doc#] where # is the index in the provided context.",
        ),
        ("human", "{input}"),
    ],
)
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_chain)


def qa():
    while True:
        q = input("🧑 > ").strip()
        if not q:
            break
        result = rag_chain.invoke({"input": q})
        print("\n🤖", result["answer"])
        print("\n----- Retrieved context -----")
        for i, doc in enumerate(result["context"], 1):
            meta = doc.metadata.get("source", "")
            print(f"[{i}] {meta} | {doc.page_content[:120]}…")
        print("-" * 40)
