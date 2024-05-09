import pickle
from typing import Any, Dict, List
import requests
import json

def load_pickle(filename: str) -> Dict[Any, Any]:
  with open(filename, 'rb') as file:
    return pickle.load(file)
  

# Tested with Mistral 7B locally with Ollama: https://ollama.com/library/mistral
def query_model(prompt):
    response_jsonl = requests.post('http://localhost:11434/api/generate', json={
        "model": "mistral",
        "prompt": prompt
    }).content.decode('utf-8').split('\n')

    response = [json.loads(r) for r in response_jsonl if r]
    array_of_tokens = [token['response'] for token in response]

    # join array of strings to get the full response
    full_response = ''.join(array_of_tokens)
    return full_response
