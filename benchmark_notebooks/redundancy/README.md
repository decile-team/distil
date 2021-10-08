#### The Effect of Redundancy

Most AL experiments assume access to a clean dataset; however, it is usually the case that data gathered "in the wild" is highly redundant. To further explore the effect of redundancy, we repeat the baseline by redundantly copying some points in CIFAR10. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![REDUNDANCY](../../experiment_plots/redundancy.png?raw=true)

We see that strategies that do not account for diversity are detrimental to the labeling efficiency of the AL loop. BADGE, which accounts for diversity, does much better than simple entropy sampling when there is an increasing amount of redundancy.