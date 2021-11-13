#### Baseline Experiments
In these experiments, we perform a comparative baseline across most of the AL algorithms in DISTIL. We utilize the ResNet18 architecture in these experiments (except for MNIST, where we instead use DISTIL's MnistNet definition), and we perform data augmentation consisting of random cropping and random horizontal flips during training. We give each strategy the same set of initial points and the same initial model. The test accuracy after training reaches 99% accuracy using SGD is reported for the corresponding labeled set size. Each experiment is repeated for a total of three times; the average and standard deviation are shown for each strategy. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![BASELINE](../../experiment_plots/baseline.png?raw=true)
![MNIST_BASELINE](../../experiment_plots/baseline_mnist.png?raw=true)

The peak labeling efficiencies in each plot show that these AL strategies can range from 1.3x to 5.0x in their labeling efficiency. Hence, AL offers benefit over random sampling in many of the common datasets used in academia.