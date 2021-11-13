#### The Role of Augmentation
In these experiments, we disable the use of data augmentation during training. We also examine the use of the VGG11 architecture. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![AUGMENT](../../experiment_plots/augmentation.png?raw=true)

Combined with a dip in test accuracy, there is also a noticeable dip in labeling efficiency when data augmentation is removed. Here, we also see that BADGE tends to outperform entropy sampling in the VGG11 architecture when data augmentation is removed, but the benefit of BADGE diminishes when data augmentation is added.