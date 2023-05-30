# bandit
バンディットアルゴリズムを色々実装してみる

## 実装済みの方策
### Contextfree
- random
- epsilon-greedy
- softmax
- UCB
### Linear
- LinUCB

# Enviroment

## Requirements
- python >=3.8.1,<4.0
- poetry

## Build
```bash
poetry install
```

# Usage

./yaml/に設定ファイルを置く.

```yaml
# yaml/sample.yaml
seed: 3090
n_trials: 15
bs: 1 # batch size
step: 5000
n_actions: 100
dim_context: 10
dim_action_context: 15
```

実行

```bash
poetry run python bin/run.py --exp-name sample
```

# Results
k: 腕の数

- k=5
![](./results/k_5.png)

- k=10
![](./results/k_10.png)

- k=50
![](./results/k_50.png)

- k=100
![](results/k_100.png)

# References
- https://github.com/st-tech/zr-obp
