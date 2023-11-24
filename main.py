import argparse
from datasets import load_dataset
from utils import set_seed, easy_print, save_results, eval_step, eval_results
from model import query_model
from attack import get_attack
from defense import get_defense
import random
import time
from defense.erase_check import IsHarmfulFilter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main evaluation')
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--data_num', default=100, type=int, help='The number of data in each split (task or inject data)')
    parser.add_argument('--model_name', default='palm2', type=str)
    parser.add_argument('--attack_method', default=None, type=str)
    parser.add_argument('--defense_method', default=None, type=str)
    parser.add_argument('--metrics', default='list', type=str)
    parser.add_argument('--result_name', default='test_results.json', type=str)
    parser.add_argument('--note', default=None, type=str)
    args = parser.parse_args()
    set_seed(args.seed)

    # AdvBench Dataset
    adv_strings = load_dataset('csv', data_files='./data/advbench/harmful_strings.csv', split='train')
    adv_behaviors = load_dataset('csv', data_files='./data/advbench/harmful_behaviors.csv', split='train')
    dataset = adv_behaviors.select(random.sample(range(len(adv_behaviors)), args.data_num))
    column_names = adv_behaviors.column_names

    if args.defense_method == 'TokenErase': 
        erase_filter = IsHarmfulFilter(model='meta-llama/Llama-2-7b-chat-hf')
        erase_mode = 'suffix' # suffix, insertion, infusion

    results = []
    for i, dp in enumerate(dataset):
        user_prompt = dp[column_names[0]]

        # 1.Now we only consider black-box jailbreaking which attacks before querying LLM
        attacked_prompt = get_attack(user_prompt, args.attack_method)

        # 2.And our defense methods are also in a preprocessing style
        if args.defense_method == 'TokenErase': 
            defensed_prompt = get_defense(attacked_prompt, args.defense_method, 
                                          erase_mode=erase_mode, 
                                          erase_filter=erase_filter,
                                          model_name=args.model_name)
        elif args.defense_method == 'RAC':
            defensed_prompt = get_defense(attacked_prompt, args.defense_method, model_name=args.model_name)
        else:
            defensed_prompt = get_defense(attacked_prompt, args.defense_method)

        # 3.When successfully defensed, it will return 'DEFENSE'
        if defensed_prompt == 'DEFENSE':
            easy_print(i, len(dataset), attacked_prompt, f'Defensed by {args.defense_method}')
            results.append({'input': attacked_prompt, 'output': f'Defensed by {args.defense_method}', 'label': 0})
            continue
        
        # 4.When fail to defense or the input prompt is safe, query LLM to get response
        output = query_model(defensed_prompt, args.model_name)
        easy_print(i, len(dataset), defensed_prompt, output)
        results.append({'input': defensed_prompt, 'output': output})
        # 也可以选择测试label
        # results.append({'input': defensed_prompt, 'output': output, 'label': eval_step(output, args.metrics)})


    # eval_results(results)
    save_results(results, args.result_name)    
