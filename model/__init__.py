# PaLM 2
import google.generativeai as palm
from google.generativeai.types import SafetySettingDict
import time
import torch
from bardapi import Bard
import requests
from transformers import pipeline


# query PaLM API
def query_palm(input: str): 

    palm_keys = [
        # Your PaLM2 keys here.
    ]

    palm.configure(api_key=palm_keys[int(time.time()) % len(palm_keys)])
    completion = palm.generate_text(
        model='models/text-bison-001',
        prompt=input,
        temperature=0,
        top_p=0.95,
        # top_k=0.1,
        max_output_tokens=2048,
        safety_settings=[
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_SEXUAL,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": palm.types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                "threshold": palm.types.HarmBlockThreshold.BLOCK_NONE,
            },             
        ],
        stop_sequences="\n"
    )
    return completion.result

def query_bard(input):
    output = bard.get_answer(input)['content']
    return output


def query_gpt(input, model_name):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': 'Bearer OpenAI API key',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 0,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = {'usage': response.json()['usage'], 'output': response.json()['choices'][0]['message']['content']}
    return result


HF_generator = None
def query_HF(input, model_path='meta-llama/Llama-2-7b-chat-hf'):
    global HF_generator
    if HF_generator == None:
        HF_generator = pipeline(
            task='text-generation', 
            model=model_path, # 'meta-llama/Llama-2-7b-chat-hf'
            device=0,
            # use_auth_token='',
            # max_length=800,
            # temperature=0,
            torch_dtype=torch.float16
            )
    completion = HF_generator(
        input,
        return_full_text=False
        )
    output = completion[0]['generated_text']  
    return output  
    

def query_model(input, model_name='palm2'):
    assert model_name in ['palm2', 'gpt-3.5-turbo', 'gpt-4', 'bard', 'llama', 'vicuna']

    if model_name == 'palm2':
        output = query_palm(input)
    elif model_name in ['gpt-3.5-turbo', 'gpt-4']:
        response = query_gpt(input, model_name)
        output = response['output']
    elif model_name == 'bard':
        output = query_bard(input)
    elif model_name == 'llama':
        output = query_HF(input)
    else:
        raise KeyError(model_name)

    return output
