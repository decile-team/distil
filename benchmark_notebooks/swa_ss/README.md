#### Generalization Techniques

Many techniques exist to help achieve better generalization when training deep models. Here, we analyze the effect that two generalization techniques have in active learning: [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) and [Shake-Shake Regularization](https://arxiv.org/abs/1705.07485). These experiments are conducted in the CIFAR-10 baseline setting with necessary modifications. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![GENERALIZATION](../../experiment_plots/generalization.png?raw=true)

Compared to the baseline setting, we see that these generalization techniques postitively contribute to the performance of AL algorithms and their labeling efficiency.