### Test suite 1 -- altering sigma
```bash
python multi_main.py \
		--run case1=experiments/configs/gmm_em_case1.yaml \
		--run case2=experiments/configs/gmm_em_case2.yaml \
    --run case3=experiments/configs/gmm_em_case3.yaml \
    --run case4=experiments/configs/gmm_em_case4.yaml \
		--trials 10 \
		--results-root experiments/results \
		--plots-root experiments/plots
```

**Case 1: close sigma**
```yaml
dgp:
  K_list: [4]
  d_R: 1
  d_X: 3
  delta_list: [0.5, 0.75, 1.0, 1.5, 2.0]
  alpha: [1.0, 2.0, 3.0, 4.0]
  sigma: [0.8, 1.0, 1.2, 1.5] # <-- Edited line
  mu_r: [-3.0, -1.0, 1.0, 3.0]
  eta0: 0.5
  eta: [1.0, -0.5, 0.8]
```
**Case 2: one large sigma**
```yaml
dgp:
  K_list: [4]
  d_R: 1
  d_X: 3
  delta_list: [0.5, 0.75, 1.0, 1.5, 2.0]
  alpha: [1.0, 2.0, 3.0, 4.0]
  # sigma: [1.0, 2.0, 4.0, 8.0]
  sigma: [1.0, 2.0, 4.0, 30.0] # <-- Edited line
  mu_r: [-3.0, -1.0, 1.0, 3.0]
  eta0: 0.5
  eta: [1.0, -0.5, 0.8]
```
**Case 3: separable sigma (default)**
```yaml
dgp:
  K_list: [4]
  d_R: 1
  d_X: 3
  delta_list: [0.5, 0.75, 1.0, 1.5, 2.0]
  alpha: [1.0, 2.0, 3.0, 4.0]
  # sigma: [1.0, 2.0, 4.0, 8.0]
  sigma: [1.0, 2.0, 4.0, 30.0] # <-- Edited line
  mu_r: [-3.0, -1.0, 1.0, 3.0]
  eta0: 0.5
  eta: [1.0, -0.5, 0.8]
```
**Case 4: small delta, close sigma**
```yaml
dgp:
  K_list: [4]
  d_R: 1
  d_X: 3
  # delta_list: [0.5, 0.75, 1.0, 1.5, 2.0]
  delta_list: [0.01, 0.2, 0.3, 0.5, 0.8, 1.5]
  alpha: [1.0, 2.0, 3.0, 4.0]
  # sigma: [1.0, 2.0, 4.0, 8.0]
  sigma: [0.8, 1.0, 1.2, 1.5]
  mu_r: [-3.0, -1.0, 1.0, 3.0]
  eta0: 0.5
  eta: [1.0, -0.5, 0.8]
```
