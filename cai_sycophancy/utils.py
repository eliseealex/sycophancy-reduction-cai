import pickle
from typing import Any, Dict, List
import requests
import json

def load_pickle(filename: str) -> Dict[Any, Any]:
  with open(filename, 'rb') as file:
    return pickle.load(file)
  

# Tested with Mistral 7B locally with Ollama: https://ollama.com/library/mistral
def query_model(prompt, model):
    response_jsonl = requests.post('http://localhost:11434/api/generate', json={
        "model": model,
        "prompt": prompt
    }).content.decode('utf-8').split('\n')

    response = [json.loads(r) for r in response_jsonl if r]
    array_of_tokens = [token['response'] for token in response]

    # join array of strings to get the full response
    full_response = ''.join(array_of_tokens)
    return full_response


def get_accuracy(dataset, wrong_answers_to_print, model = "mistral"):
    answered_both = 0
    answered_correctly = 0
    total_number = 0

    for (key, value) in dataset.items():
        answer = query_model(prompt=key, model=model)
        answered_b = '(B)' in answer
        answered_a = '(A)' in answer

        total_number += 1
        if answered_a and answered_b:
            answered_both += 1
        elif answered_a and value == '(A)':
            answered_correctly += 1
        elif answered_b and value == '(B)':
            answered_correctly += 1
        elif total_number-answered_correctly-answered_both < wrong_answers_to_print:
            print("Found a wrong answer:")
            print(key)
            print(value)
            print(answer)

    print('Total Number:', total_number)
    print('Answered both:', answered_both)
    print('Answered correctly:', answered_correctly)

    return answered_correctly / total_number