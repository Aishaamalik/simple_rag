import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chromadb

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def load_document(file):
    """Load document based on file type"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif file.name.endswith('.txt'):
        loader = TextLoader(tmp_path)
    elif file.name.endswith('.csv'):
        loader = CSVLoader(tmp_path)
    else:
        st.error("Unsupported file type. Please upload PDF, TXT, or CSV.")
        return None

    documents = loader.load()
    os.unlink(tmp_path)  # Clean up temp file
    return documents

def process_document(documents):
    """Process documents: chunk and embed"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Initialize LLM
    llm = OllamaLLM(model="llama3")

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return vectorstore, qa_chain

def main():
    st.title("Private Document Q&A App with Ollama")
    st.markdown("Upload a document and ask questions about its content.")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'csv'])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                documents = load_document(uploaded_file)
                if documents:
                    st.session_state.vectorstore, st.session_state.qa_chain = process_document(documents)
                    st.success("Document processed successfully!")

    # Q&A Interface
    if st.session_state.qa_chain is not None:
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question:")

        if st.button("Ask"):
            if question:
                with st.spinner("Generating answer..."):
                    result = st.session_state.qa_chain.invoke({"query": question})
                    st.write("**Answer:**")
                    st.write(result["result"])

                    # Show source documents
                    st.write("**Source Chunks:**")
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"Chunk {i+1}:")
                        st.write(doc.page_content[:200] + "...")
                        st.write("---")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload and process a document first.")

if __name__ == "__main__":
    main()
