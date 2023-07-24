# bandit-algorithms
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
- [rye](https://github.com/mitsuhiko/rye)

## Build
```bash
rye sync
```

# Usage
run default experiment
```bash
rye run python bin/run.py
```

multi run
```bash
rye run python bin/run.py -m n_actions=10,100,1000
```

# resources/default
k: 腕の数

- k=5
![](./resources/default/k_5.png)

- k=10
![](./resources/default/k_10.png)

- k=50
![](./resources/default/k_50.png)

- k=100
![](resources/default/k_100.png)

# References
- https://github.com/st-tech/zr-obp
