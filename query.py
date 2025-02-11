from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from models import model
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

olama= model()
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Initialize FAISS database
    db = FAISS.load_local(os.getenv('FAISS_PATH'),embeddings=olama.get_embedding_function(),allow_dangerous_deserialization=True)

    # Perform the search using the embedded query
    results = db.similarity_search_with_score(query_text, k=5)

    # Prepare the context from the documents in the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create a prompt using the context and question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model with the prompt
    model = olama.get_llm_function()
    response_text = model.invoke(prompt)

    # Extract the sources from the metadata of the results
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]  # Fallback to 'Unknown' if no ID

    # Format the final response with the response text and sources
    formatted_response = f"Response: {response_text}\nSources: {', '.join(sources)}"
   

    return response_text