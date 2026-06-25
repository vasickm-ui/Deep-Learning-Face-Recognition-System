# face_detection/app.py

import asyncio
import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embeddings import load_db, calculate_avg_embeddings
from src.recognize_frame import recognize_frame, decode_image


db = None


class UploadRequest(BaseModel):
    image: str


# load embeddings on startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db

    embeddings_db = load_db("data/vectors/vasic_embeddings.pkl")
    db = calculate_avg_embeddings(embeddings_db, 6)

    yield


app = FastAPI(lifespan=lifespan)


# enable cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
def home():
    return "Hello World"

# upload endpoint

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

    return {
        "faces": results
    }