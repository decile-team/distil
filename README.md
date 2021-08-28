<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/distil/blob/main/experiment_plots/distil_logo.svg" width="500" height="150"/>
    </br>
</p>

<p align="center">
    <a href="https://github.com/decile-team/distil/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/decile-team/distil?color=blue">
    </a>
    <a href="https://decile.org/">
        <img alt="Decile" src="https://img.shields.io/badge/website-online-green">
    </a>  
    <a href="https://decile-team-distil.readthedocs.io/en/latest/index.html">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-passing-brightgreen">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/decile-team/distil">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/decile-team/distil">
    </a>
</p>

<h3 align="center">
<p>Cut down your labeling cost and time by 3x-5x!
</h3>

# In this README
- [What is DISTIL?](#what-is-distil) 
- [Key Features of DISTIL](#key-features-of-distil)
- [Starting with DISTIL](#starting-with-distil)
- [Where can DISTIL be used?](#where-can-distil-be-used)
- [Package Requirements](#package-requirements)
- [Documentation](#documentation)
- [Make your PyTorch Model compatible with DISTIL](#make-your-pytorch-model-compatible-with-distil)
- [Demo Notebooks](#demo-notebooks)
- [Active Learning Benchmarking using DISTIL](#active-learning-benchmarking-using-distil)
- [Testing Individual Strategies and Running Examples](#testing-individual-strategies-and-running-examples)
- [Mailing List](#mailing-list)
- [Acknowledgment](#acknowledgment)
- [Team](#team)
- [Resources](#resources)
- [Publications](#publications)

## What is DISTIL?
<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/distil/blob/main/experiment_plots/distil_explanation.png" width="543" height="390"/>
    </br>
</p>

DISTIL is an active learning toolkit that implements a number of state-of-the-art active learning strategies with a particular focus for active learning in the deep learning setting. DISTIL is built on *PyTorch* and decouples the training loop from the active learning algorithm, thereby providing flexibility to the user by allowing them to control the training procedure and model. It allows users to incorporate new active learning algorithms easily with minimal changes to their existing code. DISTIL also provides support for incorporating active learning with your custom dataset and allows you to experiment on well-known datasets. We are continuously incorporating newer and better active learning selection strategies into DISTIL.

## Key Features of DISTIL
- Decouples the active learning strategy from the training loop, allowing users to modify the training and/or the active learning strategy
- Implements faster and more efficient versions of several active learning strategies
- Contains most state-of-the-art active learning algorithms
- Allows running basic experiments with just one command
- Presents interface to various active learning strategies through only a couple lines of code
- Requires only minimal changes to the configuration files to run your own experiments
- Achieves higher test accuracies with less amount of training data, admitting a huge reduction in labeling cost and time
- Requires minimal change to add it to existing training structures
- Contains recipes, tutorials, and benchmarks for all active learning algorithms on many deep learning datasets

## Starting with DISTIL

### From Git Repository
```
git clone https://github.com/decile-team/distil.git
cd distil
pip install -r requirements/requirements.txt
python train.py --config_path=/content/distil/configs/config_svhn_resnet_randomsampling.json
```
For making your custom configuration file for training, please refer to [Distil Configuration File Documentation](https://decile-team-distil.readthedocs.io/en/latest/configuration.html)

### Pip Installation
You can also install it directly as a pip package:
```python
pip install decile-distil
```

Some of the algorithms currently implemented in DISTIL include the following:

- [Uncertainty Sampling [1]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.entropy_sampling)
- [Margin Sampling [2]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.margin_sampling)
- [Least Confidence Sampling [2]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.least_confidence)
- [FASS [3]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.fass)
- [BADGE [4]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.badge)
- [GLISTER ACTIVE [6]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.glister)
- [CoreSets based Active Learning [5]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.core_set)
- [Random Sampling](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.random_sampling)
- [Submodular Sampling [3,6,7]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.submod_sampling)
- [Adversarial DeepFool [9]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.adversarial_deepfool)
- [BALD [10]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.bayesian_active_learning_disagreement_dropout)
- [Kmeans Sampling [5]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.kmeans_sampling)
- [Adversarial Bim](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.adversarial_bim)

To learn more on different active learning algorithms, check out the [Active Learning Strategies Survey Blog](https://decile-research.medium.com/active-learning-strategies-distil-62ee9fc166f9)

## Where can DISTIL be used?
DISTIL is a toolkit which provides support for various active learning algorithms. Presently, it only works in the supervised learning setting for classification. We will be adding extensions to active semi-supervised learning and active learning for object detection. It can be used in scenarios where you want to reduce labeling cost and time by labeling only the few most informative points for your ML model.

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "submodlib >= 1.1.2"

## Documentation
Learn more about DISTIL by reading our [documentation](https://decile-team-distil.readthedocs.io/en/latest/).

## Make your PyTorch Model compatible with DISTIL
DISTIL provides various models and data handlers which can be used directly.
DISTIL makes it extremely easy to integrate your custom models with active learning. There are two main things that need to be incorporated in your code before using DISTIL.

* Model
    * Your model should have a function get_embedding_dim() which returns the number of hidden units in the last layer.
    * Your forward() function should have an optional boolean parameter “last” where:
        * If True: It should return the model output and the output of the second last layer
        * If False: It should return only the model output.
    * Check the models included in DISTIL for examples!

To get a clearer idea about how to incorporate DISTIL with your own models, refer to [Getting Started With DISTIL & Active Learning Blog](https://decile-research.medium.com/getting-started-with-distil-active-learning-ba7fafdbe6f3)

## Demo Notebooks
We provide a few example notebooks using DISTIL in the notebooks folder. For ease of execution, these notebooks are written for use in Google Colab. Simply upload the selected notebook to Google Colab and connect to a hosted runtime to execute the notebook.

## Active Learning Benchmarking using DISTIL
We include a thorough benchmarking of various AL algorithms that covers many evaluation facets. Our experiments can be found in the benchmark_notebooks folder. To execute these experiments, upload a selected experiment to Google Colab and connect to a hosted runtime. We present the results of this benchmark below. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

#### Baseline Experiments
In these experiments, we perform a comparative baseline across most of the AL algorithms in DISTIL. We utilize the ResNet18 architecture in these experiments (except for MNIST, where we instead use DISTIL's MnistNet definition), and we perform data augmentation consisting of random cropping and random horizontal flips during training. We give each strategy the same set of initial points and the same initial model. The test accuracy after training reaches 99% accuracy using SGD is reported for the corresponding labeled set size. Each experiment is repeated for a total of three times; the average and standard deviation are shown for each strategy.

![BASELINE](./experiment_plots/baseline.png?raw=true)
![MNIST_BASELINE](./experiment_plots/baseline_mnist.png?raw=true)

The peak labeling efficiencies in each plot show that these AL strategies can range from 1.3x to 5.0x in their labeling efficiency. Hence, AL offers benefit over random sampling in many of the common datasets used in academia.

#### The Role of Augmentation
In these experiments, we disable the use of data augmentation during training. We also examine the use of the VGG11 architecture. 

![AUGMENT](./experiment_plots/augmentation.png?raw=true)

Combined with a dip in test accuracy, there is also a noticeable dip in labeling efficiency when data augmentation is removed. Here, we also see that BADGE tends to outperform entropy sampling in the VGG11 architecture when data augmentation is removed, but the benefit of BADGE diminishes when data augmentation is added.

#### Optimizer Effect on AL

We examine the same setting as the baseline where we instead choose to use Adam instead of SGD.

![OPTIM](./experiment_plots/adam.png?raw=true)

We see that the accuracy obtained using Adam is not as high as that obtained using SGD. Furthermore, we see that AL using SGD is able to obtain higher labeling efficiencies.

#### The Effect of Redundancy

Most AL experiments assume access to a clean dataset; however, it is usually the case that data gathered "in the wild" is highly redundant. To further explore the effect of redundancy, we repeat the baseline by redundantly copying some points in CIFAR10.

![REDUNDANCY](./experiment_plots/redundancy.png?raw=true)

We see that strategies that do not account for diversity are detrimental to the labeling efficiency of the AL loop. BADGE, which accounts for diversity, does much better than simple entropy sampling when there is an increasing amount of redundancy.

#### Active Learning versus Random Sampling

We further explore the effect of the number of examples per class in the unlabeled set on the effectiveness of AL. In these experiments, we follow the baseline setting, but we vary the number of examples in the unlabeled set on a per-class basis.

![ALVRS](./experiment_plots/alvrs.png?raw=true)

We see that, with less examples per class in the unlabeled set, the benefit of AL is reduced as there are fewer informative points to select.

#### Warm-Starting

AL practitioners may be tempted to use the working model in future AL rounds. To examine the effect of having this "warm start" model.

![RESET](./experiment_plots/persistence.png?raw=true)

Although the accuracies are close, we see that the labeling efficiency of each AL method is greater when using a model reset after each AL round.

#### More Facets

Consider reading [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324) for more evaluation facets and discussion on AL.

## Testing Individual Strategies and Running Examples
Before running the examples or test script, please clone the dataset repository in addition to this one. The default data path expects the repository in the same root directory as that of DISTIL. If you change the location, the data paths in the examples and test scripts need to be changed accordingly.

Dataset repository:
```
git clone https://github.com/decile-team/datasets.git
```

To run examples:
```
cd distil/examples
python example.py
```

To test individual strategies:
```
python test_strategy.py --strategy badge
```
For more information about the arguments that --strategy accepts:
```
python test_strategy.py -h
```
Lastly, to run all unit tests:
```
python -m unittest discover test
```

## Mailing List
To receive updates about DISTIL and to be a part of the community, join the Decile_DISTIL_Dev group.
```
https://groups.google.com/forum/#!forum/Decile_DISTIL_Dev/join 
```
## Acknowledgment
This library takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning), [Jordan Ash's Badge repository](https://github.com/JordanAsh/badge), and [Andreas Kirsch's and Joost van Amersfoort's BatchBALD repository](https://github.com/BlackHC/batchbald_redux). Also, DISTIL uses [submodlib](https://github.com/decile-team/submodlib) for submodular optimization.

## Team
DISTIL is created and maintained by Nathan Beck, [Durga Sivasubramanian](https://www.linkedin.com/in/durga-s-352831105), [Apurva Dani](https://apurvadani.github.io/index.html), [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have DISTIL more community driven. Please use it and contribute to it for your active learning research, and feel free to use it for your commercial projects. We will add the major contributors here.

## Resources
Youtube Tutorials on DISTIL:
- [Tutorial on Active Learning](https://www.youtube.com/watch?v=tBhjq1gUAv4&list=PLIQ2KoP-CQ5HU4hjT2S-HNewam8sEW-9c&index=2)
- [Tutorial and Setup of DISTIL](https://www.youtube.com/watch?v=nnNvMBUJdwc&list=PLIQ2KoP-CQ5HU4hjT2S-HNewam8sEW-9c&index=3)
- [Benchmarking Active Learning through DISTIL](https://www.youtube.com/watch?v=OGgGqk3seaw&list=PLIQ2KoP-CQ5HU4hjT2S-HNewam8sEW-9c&index=4)

[Blog Articles](https://decile-research.medium.com/)

## Publications

[1] Settles, Burr. Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009.

[2] Wang, Dan, and Yi Shang. "A new active labeling method for deep learning." 2014 International joint conference on neural networks (IJCNN). IEEE, 2014

[3] Kai Wei, Rishabh Iyer, Jeff Bilmes, Submodularity in data subset selection and active learning, International Conference on Machine Learning (ICML) 2015

[4] Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. CoRR, 2019. URL: http://arxiv.org/abs/1906.03671, arXiv:1906.03671.

[5] Sener, Ozan, and Silvio Savarese. "Active learning for convolutional neural networks: A core-set approach." ICLR 2018.

[6] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, 35th AAAI Conference on Artificial Intelligence, AAAI 2021 

[7] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[8] Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

[9] Ducoffe, Melanie, and Frederic Precioso. "Adversarial active learning for deep networks: a margin based approach." arXiv preprint arXiv:1802.09841 (2018).

[10] Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian active learning with image data." International Conference on Machine Learning. PMLR, 2017.

