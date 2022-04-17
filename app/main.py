from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time

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
async def read_item(url: str, token: str):
    received_url = 'url+"&token="+token'
    new_url = 'https://firebasestorage.googleapis.com/v0/b/learn-plus-fyp.appspot.com/o/python%2F5735505-52d1-1d20-c6c5' \
              '-aec1d1dea83-Photo%20-%20Ahmad%20Sidani.jpg?alt=media&token=0ccd9681-c6d7-4cec-9da7-4473e737b085 '
    await asyncio.sleep(2)
    return {
        "url": new_url,
        "plateNumber": 1234
    }
