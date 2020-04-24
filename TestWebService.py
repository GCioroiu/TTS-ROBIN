import requests
import json

ans = requests.request(method='GET',
                       url='http://127.0.0.1:8080/synthesis',
                       data=json.dumps({'text': 'bine ați venit la noi și dorim să vă prezentăm cea mai extraordinară aplicație.'}))

with open('response.wav', 'wb') as f:
    f.write(ans.content)