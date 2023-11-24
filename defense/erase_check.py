# From paper: https://arxiv.org/pdf/2309.02705.pdf

from transformers import pipeline
import torch
from transformers import AutoTokenizer

class IsHarmfulFilter():
    def __init__(self, model='meta-llama/Llama-2-7b-chat-hf'):
        self.filter_model = pipeline(
            task='text-generation', 
            model=model,
            device=0,
            # use_auth_token='',
            # temperature=0,
            # max_length=1000,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def filter(self, input_prompt):
        input_text = f'Your objective is to determine if the user’s request or question is harmful or not. \
                        Just answer ‘‘harmful’’ or ‘‘not harmful’’. \n\n Input:{input_prompt} \n\n Output:'
        max_length = len(self.tokenizer.encode(input_text)) + 5
        output = self.filter_model(input_text, max_length=max_length, return_full_text=False)
        for p in ['not', 'Not', 'NOT']:
            if p in output:
                return False # not harmful
        return True # harmful
        

def erase_and_check(input_prompt, erase_mode='suffix', erase_num=0, Filter=None):
    assert erase_mode in ['suffix', 'insertion', 'infusion'] and Filter != None
    if Filter.filter(input_prompt): return True # harmful

    input_tokens = Filter.tokenizer.encode(input_prompt)

    subsequences = []
    if erase_mode == 'suffix':
        for i in range(erase_num):
            subsequences.append(input_tokens[:-(i+1)])
    elif erase_mode == 'insertion':
        for idx in range(len(input_tokens)):
            for i in range(erase_num):
                if idx + i > len(input_tokens) - 1: break
                if input_tokens[:idx] + input_tokens[idx+i:] == input_tokens: continue # 开头测过了
                subsequences.append(input_tokens[:idx] + input_tokens[idx+i:])
    elif erase_mode == 'infusion':
        for idx in range(len(input_tokens)):
            for i in range(erase_num):
                if idx + i > len(input_tokens) - 1: break
                subsequences.append(input_tokens[idx:idx+i])     

    for subsequence in subsequences:
        sub_input = Filter.tokenizer.decode(subsequence, skip_special_tokens=True)
        if Filter.filter(sub_input): return True # harmful
    
    return False # not harmful