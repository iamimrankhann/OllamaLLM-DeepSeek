from langchain_ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

class model:
    def get_embedding_function(self):
        """
        Returns the Ollama embedding function for text embedding.
        """
        # embedding = ollama.embeddings("nomic-embed-text")  # Using Ollama embedding model
        embedding= OllamaEmbeddings(model="nomic-embed-text")
        return embedding
    def get_llm_function(self):
        """
        Returns the Ollama llm function for answer the query.
        """
        llm = Ollama(model="deepseek-r1:1.5b")
        return llm