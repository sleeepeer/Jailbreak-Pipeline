# LLMGuard: A simple jailbreaking LLM pipeline

## Setup

```bash
conda env create -f environment.yml
conda activate jailbreak
```

## Quick Usage

* First, modify run config in **run.py**.

```python
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
```

* Then run this code.

```bash
python run.py
```

* If you want to evaluate the ASR, please use **eval.ipynb** or **eval_step()** in **utils** folder. I implemented some eval metrics there.

