#### Data Subset Selection

Recent efforts in deep learning have applied data subset selection techniques to accelerate training while retaining high accuracy. Here, we analyze the effect of performing some of the training rounds using DSS techniques such as [GradMatch](https://arxiv.org/abs/2103.00123). For comparison, we do full training on select rounds. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324). 

| DSS | 1000 points | 10000 points | 19000 points | 25000 points |
| :--- | :----: | :----: | :----: | :----: |
Random (GM) | 53.7 +- 0.0 | 85.2 +- 0.4 | 89.7 +- 0.3 | 90.5 +- 0.5 |
Random (F) | 53.7 +- 0.0 | 85.1 +- 0.9 | 89.7 +- 0.4 | 90.5 +- 0.6 |
Entropy (GM) | 53.7 +- 0.0 | 88.5 +- 0.1 | 92.6 +- 0.4 | 93.5 +- 0.5 |
Entropy (F) | 53.7 +- 0.0 | 89.0 +- 0.3 | 92.6 +- 0.1 | 93.6 +- 0.2 |
Badge (GM) | 53.7 +- 0.0 | 87.9 +- 0.5 | 92.2 +- 0.6 | 93.3 +- 0.6 |
Badge (F) | 53.7 +- 0.0 | 88.7 +- 0.1 | 92.3 +- 0.4 | 93.4 +- 0.4 |

We see that DSS techniques have the potential to speed up AL rounds while maintaining good performance.