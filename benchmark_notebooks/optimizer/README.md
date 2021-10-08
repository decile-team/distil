#### Optimizer Effect on AL

We examine the same setting as the baseline where we instead choose to use Adam instead of SGD.More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![OPTIM](../../experiment_plots/adam.png?raw=true)

We see that the accuracy obtained using Adam is not as high as that obtained using SGD. Furthermore, we see that AL using SGD is able to obtain higher labeling efficiencies.