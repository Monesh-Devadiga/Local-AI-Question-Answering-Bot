from langchain_ollama import ChatOllama
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from ingest import load_vectorstore, CHROMA_DIR

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the "
    "following context. If the context does not contain enough information, "
    "say so honestly. Be concise and accurate.\n\n"
    "Context:\n{context}"
)

AVAILABLE_MODELS = ["mistral", "llama3", "gemma", "phi3"]


def get_rag_chain(
    model_name: str = "mistral",
    chroma_dir: str = CHROMA_DIR,
    embedding_model: str = "nomic-embed-text",
):
    vectorstore = load_vectorstore(chroma_dir=chroma_dir, embedding_model=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model=model_name, temperature=0.3)

    # Prompt for context-aware follow-up questions
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rephrase it "
         "as a standalone question that can be answered using the provided context. "
         "Do NOT answer the question, just reformulate it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Main QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_invoke(user_input: str, chat_history):
        # Convert chat_history list of (human, ai) tuples to LangChain messages
        lc_history = []
        for human, ai in chat_history:
            lc_history.append(HumanMessage(content=human))
            lc_history.append(AIMessage(content=ai))

        # Retrieve relevant docs
        retrieved_docs = history_aware_retriever.invoke({
            "input": user_input,
            "chat_history": lc_history,
        })

        # Format context
        context = format_docs(retrieved_docs)

        # Generate answer
        chain = qa_prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "chat_history": lc_history,
            "input": user_input,
        })
        return response

    return rag_invoke
