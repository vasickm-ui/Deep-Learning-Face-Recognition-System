import asyncio
import base64
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embeddings import load_db, calculate_avg_embeddings
from src.recognize_frame import recognize_frame, decode_image


APP_ENV = "development"

if APP_ENV == "production":
    load_dotenv(".env.production")
else:
    load_dotenv(".env.development")

db = None


class UploadRequest(BaseModel):
    image: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db

    embeddings_db = load_db("data/vectors/vasic_embeddings.pkl")
    db = calculate_avg_embeddings(embeddings_db, 6)

    yield


app = FastAPI(lifespan=lifespan)

if APP_ENV == "development":
    allow_origins = ["http://localhost:3000"]
else:
    allow_origins = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/test")
def test():
    return {"msg": "radi", "env": APP_ENV}


@app.post("/upload")
async def upload(request: UploadRequest):
    frame = decode_image(request.image)

    loop = asyncio.get_running_loop()

    results = await loop.run_in_executor(
        None,
        recognize_frame,
        frame,
        db
    )

    return {"faces": results}