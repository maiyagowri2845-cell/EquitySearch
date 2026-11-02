import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

st.set_page_config(
    page_title="EquitySearch",
    page_icon="📈",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📈 EquitySearch")
st.caption("AI-powered financial news research assistant")
st.sidebar.title("📰 News Articles")
st.sidebar.caption("Enter up to 3 article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url.strip())
if len(urls) != len(set(urls)):
    st.sidebar.warning("⚠️ Duplicate URLs detected.")
    urls = list(dict.fromkeys(urls))
urls = list(dict.fromkeys(urls))

process_url_clicked = st.sidebar.button("Process URLs")
comparison_mode = st.sidebar.checkbox("📊 Comparison Mode",help="Compare information across the indexed articles.")
if process_url_clicked and not urls:
    st.sidebar.error("Please enter at least one article URL.")
    st.stop()
if process_url_clicked and comparison_mode and len(urls) < 2:
    st.sidebar.error("📊 Comparison Mode requires at least 2 article URLs.")
    st.stop()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

file_path = "faiss_store"

main_placeholder = st.empty()
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

if process_url_clicked:

    try:
        with st.spinner("📥 Loading articles..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

        if not data:
            st.error("❌ No articles could be loaded. Please check your URLs.")
            st.stop()

        st.success(f"✅ Loaded {len(data)} article(s)")
        st.subheader("📋 Article Summaries")

        for i, article in enumerate(data, start=1):
            with st.expander(f"Article {i} Summary", expanded=False):
                summary_prompt = ChatPromptTemplate.from_template(
                    """Summarize the following financial news article.

        Focus on:
        - Main event
        - Companies or stocks mentioned
        - Important financial figures
        - Potential market impact

        Keep the summary concise and factual.

        Article:
        {article}
        """
                )

                summary_chain = summary_prompt | llm

                summary = summary_chain.invoke({
                    "article": article.page_content[:12000]
                }).content

                st.write(summary)

        with st.spinner("✂️ Splitting articles into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)
            for doc in docs:
                doc.metadata["article_url"] = doc.metadata.get(
                    "source",
                    "Unknown source"
                )

        st.success(f"✅ Created {len(docs)} text chunks")
        st.subheader("📊 Research Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Articles Loaded", len(data))
        with col2:
            st.metric("Text Chunks", len(docs))
        with col3:
            st.metric(
                "Comparison Mode",
                "ON" if comparison_mode else "OFF"
            )

        with st.spinner("🧠 Creating embeddings and building FAISS index..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(file_path)
            st.session_state.chat_history = []

        st.success("✅ Articles indexed successfully!")
        st.info(
            f"📊 EquitySearch indexed {len(data)} article(s) "
            f"and created {len(docs)} searchable chunks."
        )
        st.info(f"🔎 Ready to search across {len(urls)} article(s).")

    except Exception as e:
        st.error("❌ Something went wrong while processing the articles.")
        st.exception(e)

st.subheader("💬 Research Chat")
if comparison_mode:
    st.caption("📊 Comparison Mode: analyzing articles separately")
else:
    st.caption("🔎 Research Mode: searching across all indexed articles")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask a question about the articles...")
if query:
    if not os.path.exists(file_path):
        st.warning("⚠️ Please process at least one article before asking a question.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    contextualize_prompt = ChatPromptTemplate.from_template(
            """Given the conversation history and the latest user question,
        rewrite the latest question so it can be understood independently.

        If the question is already standalone, return it unchanged.

        Conversation history:
        {chat_history}

        Latest question:
        {input}

        Standalone question:"""
    )

    contextualize_chain = contextualize_prompt | llm

    chat_history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in st.session_state.chat_history
    )

    if chat_history_text:
        standalone_question = contextualize_chain.invoke({
            "chat_history": chat_history_text,
            "input": query
        }).content
    else:
        standalone_question = query
    if comparison_mode:
        if len(urls) < 2:
            st.warning(
                "📊 Comparison Mode requires at least 2 article URLs."
            )
            st.stop()
        comparison_docs = []
        for url in urls:
            docs_for_article = vectorstore.similarity_search(
                standalone_question,
                k=2,
                filter={"article_url": url}
            )

            comparison_docs.extend(docs_for_article)
        if not comparison_docs:
            st.warning("⚠️ No relevant information was found in the indexed articles.")
            st.stop()
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {input}

        If the answer cannot be found in the context, say that you don't know.
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    if comparison_mode:
        answer = document_chain.invoke({
            "context": comparison_docs,
            "input": standalone_question
        })

        retrieved_docs = comparison_docs

    else:
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        result = retrieval_chain.invoke({
            "input": standalone_question
        })

        answer = result["answer"]
        retrieved_docs = result["context"]

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(answer)

        # Display sources, if available
    st.subheader(f"📚 Sources ({len(retrieved_docs)} retrieved)")

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown source")

        with st.expander(f"Source {i}: {source}"):
            st.write(doc.page_content)
            st.caption(f"Source URL: {source}")