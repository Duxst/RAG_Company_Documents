import os
import streamlit as st
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import numpy as np
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings  # Added missing import
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  # Updated import to use PdfReader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import c# Initialize Pinecone, Groq, and embedding model
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.colab import userdata
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import tiktoken
import os
from groq import Groq

# Set up API keys and other configurations using st.secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]  # Retrieve from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]          # Retrieve from secrets.toml
index_name = "rag-workshop"
namespace = "company-documents"

# Utility function to extract text from PDF (using PyPDF2)
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Streamlit Interface Setup
st.title("Upload Your PDF and Ask Questions")
st.write("This interface allows you to upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize Pinecone, Groq, and embedding model
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)

@st.cache_resource
def init_groq():
    return Groq(api_key=groq_api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

pinecone_index = init_pinecone()
groq_client = init_groq()
embedding_model = load_embedding_model()

# Step 1: Document Loading Function
def load_document(pdf_file):
    # Use the function `extract_text_from_pdf` which is now defined above
    document_content = extract_text_from_pdf(pdf_file)
    document_source = pdf_file.name
    
    doc = Document(
        metadata={"source": document_source},
        page_content=f"Source: {document_source}\n{document_content}"
    )
    return doc

# Step 2: Store Document in Vector Store
def store_document(doc, embedding_model):
    embeddings = embedding_model.encode([doc.page_content])
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=HuggingFaceEmbeddings(embedding_model))
    vectorstore_from_documents = PineconeVectorStore.from_documents(
        [doc],
        HuggingFaceEmbeddings(embedding_model),
        index_name=index_name,
        namespace=namespace
    )

# Step 3: Perform RAG Query
def perform_rag(query):
    raw_query_embedding = embedding_model.encode([query])
    query_embedding = np.array(raw_query_embedding)

    # Query Pinecone
    top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

    # Prepare contexts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n" + query

    # Create system prompt
    system_prompt = """
    You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, inventory reports, and postal codes.
    Answer any questions I have, based on the data provided. Always consider all parts of the context provided when forming a response.
    """

    # Make Groq request
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return response.choices[0].message.content

# Interface Actions
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        document = load_document(uploaded_file)
        st.success("PDF loaded successfully!")
        
        # Store document in Pinecone
        store_document(document, embedding_model)
        st.success("Document indexed successfully!")

    # Input Box for Query
    query = st.text_input("Ask a question about the uploaded PDF:")
    if query:
        with st.spinner("Generating response..."):
            response = perform_rag(query)
            st.write(response)
