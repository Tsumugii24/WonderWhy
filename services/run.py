import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))

import uvicorn
from fastapi import FastAPI

from utils import set_api_key_from_json
from rag import rag_router

app = FastAPI()
# Include routers in the main app
app.include_router(rag_router, prefix="/v1/kb", tags = ["RAG Services"])

# home page
@app.get("/")
async def welcome() -> dict:
    return {"message": "Welcome to lic rag survices"}


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_json_path = os.path.join(ROOT_DIR, "config.json")
    set_api_key_from_json(cfg_json_path)

    uvicorn.run(app="run:app", host="0.0.0.0", port=8000, reload=True)