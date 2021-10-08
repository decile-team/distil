#### Effect of AL Budget

Different choices of the AL budget are frequently made across AL literature. Here, we examine if reasonable choices of the budget have noticeable effects in the CIFAR-10 baseline setting. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

| Alg_Budget | 1000 points | 7000 points | 13000 points | 19000 points | 25000 points | 
| :--- | :----: | :----: | :----: | :----: | :----: |
| Rand_1000 | 55.6 +- 0.0 | 82.5 +- 0.5 | 87.8 +- 0.6 | 89.1 +- 0.8 | 91.0 +- 0.2 |
| Rand_3000 | 55.6 +- 0.0 | 82.2 +- 1.2 | 87.0 +- 0.2 | 89.9 +- 0.5 | 90.5 +- 0.5 |
| Rand_6000 | 55.6 +- 0.0 | 82.0 +- 0.8 | 87.7 +- 0.7 | 89.4 +- 0.7 | 90.7 +- 0.6 | 
| Ent_1000 | 55.6 +- 0.0 | 84.3 +- 0.9 | 88.6 +- 1.2 | 91.9 +- 0.9 | 92.8 +- 0.8 |
| Ent_3000 | 55.6 +- 0.0 | 83.3 +- 1.4 | 89.4 +- 1.2 | 91.7 +- 0.8 | 92.5 +- 0.8 |
| Ent_6000 | 55.6 +- 0.0 | 82.7 +- 1.0 | 88.5 +- 0.1 | 91.0 +- 1.1 | 92.4 +- 0.8 |
| Badge_1000 | 55.6 +- 0.0 | 83.5 +- 1.5 | 89.0 +- 0.6 | 90.8 +- 1.2 | 92.4 +- 0.6 |
| Badge_3000 | 55.6 +- 0.0 | 83.4 +- 1.3 | 88.4 +- 0.9 | 91.5 +- 0.7 | 92.6 +- 0.5 |
| Badge_6000 | 55.6 +- 0.0 | 82.13 +- 1.4 | 88.1 +- 0.6 | 91.1 +- 0.9 | 92.3 +- 1.1 |

We find that reasonable choices of the batch size make little difference in the achieved test accuracies in this setting.