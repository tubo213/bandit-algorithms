# bandit-algorithms
バンディットアルゴリズムを色々実装してみる

## 実装済みの方策

### Default

#### Contextfree
- random
- epsilon-greedy
- softmax
- UCB
#### Linear
- LinUCB

### Multiple-Play Bandit Problem

#### Contextfree
- random
- Multiple-Play Epsilon-Greedy
- Multiple-Play UCB
- Multiple-Play Thompson Sampling
- PBM-UCB
- PBM-PIE

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

# Results

## Default
k: 腕の数

- k=5
![](./resources/default/n_actions=5.png)

- k=25
![](./resources/default/n_actions=25.png)

- k=125
![](./resources/default/n_actions=125.png)

- k=625
![](resources/default/n_actions=625.png)

### Multiple-Play Bandit Problem

#### Position Based Model(Contextfree)
k: 腕の数
L: プレイ回数 = 10

腕の数が多くなるほどPBMベースの方策が良い結果を出すが、Epsilon-Greedyが強くなる

- k=15
![](./resources/pbm/n_actions=15.png)

- k=30
![](./resources/pbm/n_actions=30.png)

- k=60
![](./resources/pbm/n_actions=60.png)

- k=120
![](./resources/pbm/n_actions=120.png)

- k=240
![](./resources/pbm/n_actions=240.png)

- k=480
![](./resources/pbm/n_actions=480.png)

- k=960
![](./resources/pbm/n_actions=960.png)

# References
- https://github.com/st-tech/zr-obp
