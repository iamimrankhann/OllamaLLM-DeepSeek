from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from models import model
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
class RAG:

    def pdfload(self,path):
        """
        Function to load all PDF files from a given directory.
        
        Args:
            path (str): Path to the directory containing PDFs.
        
        Returns:
            list: List of Document objects containing text extracted from PDFs.
        """
        my_doc = PyMuPDFLoader(path)  # Load all PDFs in the given directory
        return my_doc.load()  # Extracts text from all PDFs and returns a list of documents

    def split_documents(self,documents: list[Document]):
        """
        Function to split large text documents into smaller chunks.
        
        Args:
            documents (list[Document]): List of Document objects containing text.
        
        Returns:
            list: List of smaller Document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # Defines maximum chunk size (500 characters)
            chunk_overlap=70,     # Defines overlap between consecutive chunks (70 characters)
            length_function=len,  # Uses Python's `len()` function to measure chunk size
            is_separator_regex=False,  # Indicates that separator is a string, not a regex pattern
        )
        return text_splitter.split_documents(documents)  # Returns a list of chunked documents


    def calculate_chunk_ids(self,chunks):
        """
        Calculates and adds chunk IDs based on the page number and chunk index.
        
        Args:
            chunks (list[Document]): List of Document chunks.
        
        Returns:
            list[Document]: Updated chunks with unique IDs.
        """
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the chunk's metadata.
            chunk.metadata["id"] = chunk_id

        return chunks

    def add_to_faiss(self,chunks: list[Document]):
        """
        Adds new document chunks to the FAISS vector database, if not already present.
        
        Args:
            chunks (list[Document]): List of document chunks to be added.
        """
        models=model()
        # Create a FAISS vector store (you can persist it locally if needed)
        db = FAISS.from_documents(chunks, models.get_embedding_function())

        # Calculate chunk IDs
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Retrieve existing items from FAISS (if any, depending on your use case)
        existing_ids = set()
        

        print(f"Number of existing documents in FAISS DB: {len(existing_ids)}")

        # Only add new documents that don't exist in the FAISS DB
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            # Add the new chunks to FAISS
            db.add_documents(new_chunks)  # FAISS adds vectors, so ensure embeddings are included
            # If you need to persist the FAISS index
            db.save_local(os.getenv("FAISS_PATH"))
        else:
            print("✅ No new documents to add")
        return 'success'