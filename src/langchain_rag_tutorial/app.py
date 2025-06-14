from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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

rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question so it makes sense without the chat history.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ],
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    rephrase_prompt,
)

stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

rag_logic = create_retrieval_chain(
    history_aware_retriever,
    stuff_chain,
)

rag_chain = RunnableWithMessageHistory(
    rag_logic,
    lambda _: InMemoryChatMessageHistory(),  # session_id -> 空の履歴を返す関数
    input_messages_key="input",  # 人間の発話が入るキー
    history_messages_key="chat_history",  # 履歴を渡すキー
    output_messages_key="answer",  # AI の返答が入るキー
)


def qa():
    session_id = "cli"
    while True:
        q = input("🧑 > ").strip()
        if not q:
            break

        result = rag_chain.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}},
        )

        print("\n🤖", result["answer"])
        print("\n----- Retrieved context -----")
        for i, doc in enumerate(result["context"], 1):
            src = doc.metadata.get("source", "")
            print(f"[{i}] {src} | {doc.page_content[:120]}…")
        print("-" * 40)
