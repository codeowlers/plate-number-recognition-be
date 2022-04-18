from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .script import number_plate_recognition

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/run")
async def read_item(epochs: int, url: str, token: str):
    received_url = str(url) + "&token=" + token
    return number_plate_recognition(received_url, epochs)
