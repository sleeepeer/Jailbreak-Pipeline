import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"nohup python3 -u main.py \
        --seed {test_params['seed']}\
        --data_num {test_params['data_num']}\
        --model_name {test_params['model_name']}\
        --attack_method {test_params['attack_method']}\
        --defense_method {test_params['defense_method']}\
        --metrics {test_params['metrics']}\
        --result_name {log_name + '.json'}\
        --note {test_params['note']}\
        > {log_file} &"
        
    os.system(cmd)


def get_log_name(test_param):
    # Generate a log file name
    log_dir = './logs'
    os.makedirs(f'{log_dir}', exist_ok=True)

    log_name = f"{test_param['model_name']}-attack:{test_param['attack_method']}-defense:{test_param['defense_method']}-data_num:{test_param['data_num']}-seed:{test_param['seed']}"

    if test_param['note'] != None:
        log_name = f"{test_param['note']}" + log_name
    
    file_name = log_name + '.txt'
    # os.makedirs(f'{log_dir}/{log_name}', exist_ok=True)

    return f'{log_dir}/{file_name}', log_name


test_params = {
    'seed': 23,
    'data_num': 100,
    'model_name': 'palm2',           # palm2, gpt-3.5-turbo, bard
    'attack_method': 'OJA',          # HJA, OJA, Fuzzing,
    'defense_method': 'RAC',         # Moderation, TokenErase, RAC
    'metrics': 'classifier',         # list, classifier
    'note': None
}
run(test_params)