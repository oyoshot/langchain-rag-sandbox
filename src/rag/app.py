from collections import defaultdict

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.util import VECTOR_DIR

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
            "Answer using the following documents when relevant:\n{context}\n\n"
            "If you have already enumerated items earlier in the chat, keep your count consistent "
            "unless new documents introduce additional items.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ],
)
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

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

rag_logic = create_retrieval_chain(
    history_aware_retriever,
    stuff_chain,
)

_histories: dict[str, InMemoryChatMessageHistory] = defaultdict(
    InMemoryChatMessageHistory,
)

rag_chain = RunnableWithMessageHistory(
    rag_logic,
    lambda session_id: _histories[session_id],
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
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
