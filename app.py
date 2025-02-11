from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
from rag import RAG
import query
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

UPLOAD_DIRECTORY = os.getenv("FILE_UPLOAD")
if os.path.exists(UPLOAD_DIRECTORY):
    shutil.rmtree(UPLOAD_DIRECTORY)
os.makedirs(UPLOAD_DIRECTORY)

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Define the file path
        global file_path
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"status":200,"filename": file.filename, "filepath": file_path}
    
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the file: {str(e)}")
@app.post("/trainsLLM/")
async def train_llm():
    pipeline = RAG()
    mypdf = pipeline.pdfload(file_path)
    print("pdf done upload")
    mychunks = pipeline.split_documents(mypdf)
    print("chunk splited done")
    faiss_update = pipeline.add_to_faiss(mychunks)
    print("fais update done")
    if faiss_update=='success':
        return {
            "status":200,
            "message": "LLM model trained successfully on given pdf!!"
            }
@app.post("/UserQuery/")
async def query_text_user(query_text:str):
    response  = query.query_rag(query_text)
    return{
        "status":200,
        "response":response
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

        

