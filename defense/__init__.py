import openai
from .erase_check import erase_and_check
from .RAC import RAC_defense
openai.api_key = 'your key here'


def get_defense(input_prompt, defense_method=None, **kwargs):
    if defense_method == None or defense_method == 'None':
        return input_prompt
    elif defense_method == 'Moderation':
        response = openai.Moderation.create(input=input_prompt)
        if response["results"][0]['flagged']:
            return 'DEFENSE'
        else: return input_prompt
    elif defense_method == 'TokenErase':
        erase_mode = kwargs.get('erase_mode', None)
        erase_filter = kwargs.get('erase_filter', None)
        if erase_and_check(input_prompt, erase_mode, 6, erase_filter): 
            return 'DEFENSE'
        else: return input_prompt
    elif defense_method == 'RAC':
        model_name = kwargs.get('model_name', None)
        # print(model_name)
        return RAC_defense(input_prompt, model_name)