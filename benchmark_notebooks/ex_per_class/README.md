#### Active Learning versus Random Sampling

We further explore the effect of the number of examples per class in the unlabeled set on the effectiveness of AL. In these experiments, we follow the baseline setting, but we vary the number of examples in the unlabeled set on a per-class basis. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![ALVRS](../../experiment_plots/alvrs.png?raw=true)

We see that, with less examples per class in the unlabeled set, the benefit of AL is reduced as there are fewer informative points to select.