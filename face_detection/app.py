import asyncio
import base64
import os
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embeddings import load_db, calculate_avg_embeddings
from src.recognize_frame import recognize_frame, decode_image
from src.paths import PROJECT_ROOT


APP_ENV = "development"

if APP_ENV == "production":
    load_dotenv(".env.production")
else:
    load_dotenv(".env.development")

db = None
DB_PATH = PROJECT_ROOT / "data" / "vectors" / "vasic_embeddings.pkl"
print(DB_PATH)

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
    allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
else:
    allow_origins = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*", "OPTIONS"],
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

    print("REZULTATI")
    print(results)

    return jsonable_encoder(results)