from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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
    received_url = url+"&token="+token
    new_url = 'https://i.pinimg.com/originals/f7/2b/ed/f72bed70dc454c8959c5c0b74df13638.jpg'
    await asyncio.sleep(10)
    return {
        "url": new_url,
        "plateNumber": 1234,
        "received_url": received_url
    }
