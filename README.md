# Refractory AdamW
When a parameter updates significantly, its refractory state increases, reducing subsequent updates. The state decays exponentially back to zero.

## Results

### nanoGPT

```
Baseline: step 2000: train loss 1.7648, val loss 1.8857
Refractory: step 2000: train loss 1.7561, val loss 1.8834
```

### nanoRWkV

```
Baseline: step 2000: train loss 1.4449, val loss 1.6490
Refractory: step 2000: train loss 1.4426, val loss 1.6435
```

### Cartpole

```
n_episode :380, score : 3206.5, n_buffer : 50000, eps : 6.1%
```
