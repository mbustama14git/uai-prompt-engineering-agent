import requests
import json 
import time



chat = {"message": {"text": "", "chat": {"id": 1}}, "type": 1}
url = "http://127.0.0.1:8000/messages"



headers = {
  'Content-Type': 'application/json',
  'token': "..438OJ7gwaEinsdAw0OsP303jQqKEI01ultxscV2-dok"
}

def input_user():
    print("\n")
    print("-- Texto usuario: ")
    texto = input()
    print("\n")
    return texto


while True:
    texto = input_user()
    # envio de mensajes
    chat['message']["text"] = texto
    payload = json.dumps(chat)
    # tiempo inicial
    tiempo_inicial = time.time()

    response = requests.post(url, headers=headers, data=payload)
    result = response.json()
    #print(response.status_code)
    print(result['response'])