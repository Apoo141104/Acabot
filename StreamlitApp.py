import streamlit as st
from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception  # Assuming this is a custom exception handler
from logger import logging  # Assuming this is a custom logging module
from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
#from llama_index.core.settings import Settings
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")  # Create the generative model

import tempfile

def load_data(file):
    """
    Load PDF documents from the uploaded file.

    Parameters:
    - file (UploadedFile): The file uploaded through Streamlit.

    Returns:
    - A list of loaded PDF documents (content extracted from PDFs).
    """
    try:
        logging.info("Data loading started...")
        # Create a temporary directory to save the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            # Use the temporary directory path to load data
            loader = SimpleDirectoryReader(temp_dir)
            documents = loader.load_data()
        logging.info("Data loading completed...")
        return documents
    except Exception as e:
        logging.info("Exception in loading data...")
        raise customexception(e, sys) 

# ... other imports

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model,
    generates document embeddings, and creates a VectorStoreIndex.

    Returns:
    - Tuple: (query_engine, embeddings) - query engine for querying, embedding list
    """
    try:
        logging.info("")

        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        # No need for system_prompt attribute

        service_context = ServiceContext.from_defaults(llm=model,embed_model=gemini_embed_model, chunk_size=800, chunk_overlap=20)

        logging.info("")
        with StorageContext.from_defaults():
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            index.storage_context.persist()

        logging.info("")
        query_engine = index.as_query_engine()

        # Generate document embeddings
        embeddings = gemini_embed_model.embed_documents(documents)

        return query_engine, embeddings

    except Exception as e:
        # Handle potential nested exceptions from 'customexception'
        logging.error("Error in download_gemini_embedding:", exc_info=True)
        raise customexception(e, sys)  # Re-raise the exception


def main():
    st.set_page_config("QA with Documents")

    doc = st.file_uploader("Upload any document")

    st.header("Acabot-Your Personal Academic Companion")

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if doc is not None:  # Check if a document is uploaded
                try:
                    document = load_data(doc)
                    query_engine, embeddings = download_gemini_embedding(model, document)

                    response = query_engine.query(user_question)
                    st.write(response.response)

                    # Optionally display or utilize the document embeddings (example):
                    st.write("Example Document Embeddings (first 5):")
                    for i, embedding in enumerate(embeddings[:5]):
                        st.write(f"Document {i+1}: {embedding}")
                except Exception as e:
                    logging.error("Error in main function:", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")  # Provide informative error message
            else:
                st.error("Please upload a document to process.")

if __name__ == "__main__":
    main()
