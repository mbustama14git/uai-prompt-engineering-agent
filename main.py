from openai import OpenAI 
import os
import requests
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from bd.vector import find_vector_in_redis
from ai.chat import generate_text
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import config



client = OpenAI(api_key=config.gpt_key)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_FIELD_NAME = 'content_vector'

@app.get("/")
async def init():
    return "hi"

@app.post("/messages")
def messages(payload: dict = Body(...)):
    chat_id = payload['message']['chat']['id']
    typeApp = payload['type']
    message = payload['message']['text']
    try:
        prompt = f"pregunta: {message}"
        response = generate_text(prompt, chat_id)
        return {"response": response}  # <-- Retorna la respuesta de la IA

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={"error": "Error interno del servidor"}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



