#### Seed Set Initialization

Some AL initializations use carefully constructed seed sets. Here, we compare the performance of AL when carefully constructed seed sets are used versus randomly constructed seed sets. The experiments are conducted in the CIFAR-10 baseline setting. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324). 

| Alg_Init | 1000 points | 4000 points | 7000 points | 10000 points | 25000 points |
| :---     | :----:      | :----:      | :----:      | :----:       | :----:       | 	
|Ent_FL| 57.8 +- 0.0 | 78.6 +- 0.7 | 85.8 +- 0.8 | 88.2 +- 0.8 | 93.5 +- 0.2 |
|Ent_Rand | 53.7 +- 0.0 | 79.1 +- 0.3 | 85.9 +- 0.4 | 89.0 +- 0.3 | 93.6 +- 0.2 |
|Badge_FL | 57.8 +- 0.0 | 77.9 +- 0.3 | 85.0 +- 0.7 | 88.5 +- 0.5 | 93.2 +- 0.4 |
|Badge_Rand | 53.8 +- 0.0 | 78.1 +- 0.7 | 84.9 +- 0.4 | 88.6 +- 0.1 | 93.4 +- 0.4 |

We see that the initial benefit of the carefully constructed seed sets (via facility location maximization) vanishes after only a few rounds.