# bandit
banditを色々実装してみる

# Usage

./yaml/に設定ファイルを置く.

```yaml
# yaml/sample.yaml
n_actions: 2
```

実行

```bash
poetry run python main.py --exp-name sample
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


