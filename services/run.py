import uvicorn
from fastapi import FastAPI

from rag import rag_router

app = FastAPI()

# Include routers in the main app
app.include_router(rag_router, prefix="/v1/kb", tags = ["RAG Services"])



# home page
@app.get("/")
async def welcome() -> dict:
    return {"message": "Welcome to lic rag survices"}


if __name__ == "__main__":
    uvicorn.run(app="run:app", host="0.0.0.0", port=8000, reload=True)
