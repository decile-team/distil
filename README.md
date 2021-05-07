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
- [Installation](#installation)
- [Package Requirements](#package-requirements)
- [Documentation](#documentation)
- [Make your PyTorch Model compatible with DISTIL](#make-your-pytorch-model-compatible-with-distil)
- [Demo Notebooks](#demo-notebooks)
- [Active Learning Benchmarking using DISTIL](#active-learning-benchmarking-using-distil)
- [Testing Individual Strategies and Running Examples](#testing-individual-strategies-and-running-examples)
- [Mailing List](#mailing-list)
- [Acknowledgement](#acknowledgement)
- [Team](#team)
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
```
git clone https://github.com/decile-team/distil.git
cd distil
python train.py --config_path=/content/distil/configs/config_svhn_resnet_randomsampling.json
```
For making your custom configuration file for training, please refer to [Distil Configuration File Documentation](https://decile-team-distil.readthedocs.io/en/latest/configuration.html)

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
- [Baseline Sampling](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.baseline_sampling)

To learn more on different active learning algorithms, check out the [Active Learning Strategies Survey Blog](https://decile-research.medium.com/active-learning-strategies-distil-62ee9fc166f9)

## Where can DISTIL be used?
DISTIL is a toolkit which provides support for various active learning algorithms. Presently, it only works in the supervised learning setting for classification. We will be adding extensions to active semi-supervised learning and active learning for object detection. It can be used in scenarios where you want to reduce labeling cost and time by labeling only the few most informative points for your ML model.

## Installation
The latest version of  DISTIL package can be installed using the following command:

```python
pip install --extra-index-url https://test.pypi.org/simple/ decile-distil
```
### NOTE
```
Please make sure to enter the space between simple/ and decile-distil in the above command while installing the DISTIL package!
```

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "apricot-select >= 0.6.0"

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

* Data Handler
    * Your DataHandler class should have a boolean attribute “select=True” with default value True:
        * If True: Your __getitem__(self, index) method should return (input, index)
        * If False: Your __getitem__(self, index) method should return (input, label, index)
    * Your DataHandler class should have a boolean attribute “use_test_transform=False” with default value False.
    
    * Check the DataHandler classes included in DISTIL for examples!

To get a clearer idea about how to incorporate DISTIL with your own models, refer to [Getting Started With DISTIL & Active Learning Blog](https://decile-research.medium.com/getting-started-with-distil-active-learning-ba7fafdbe6f3)

## Demo Notebooks
1. [CIFAR10 Tutorial](https://colab.research.google.com/drive/1K5eFLtJYbNEpDRI6YsYFCEQyi74tfEou?usp=sharing)

2. [SATIMAGE Tutorial](https://colab.research.google.com/drive/1wD-so8B14kSswZwCDHDhLGkkGIH7VUdU?usp=sharing)

3. [IJCNN1 Tutorial](https://colab.research.google.com/drive/11WRz7CXC4ZCvmEkXQ-FQ2YiwRZY7QIof?usp=sharing)

You can also download the .ipynb files from the notebooks folder.

## Active Learning Benchmarking using DISTIL
#### Experimentation Method
The models used below were first trained on an initial random set of points (equal to the budget). For each set of new points added, the model was trained from scratch until the training accuracy crossed the max accuracy threshold. The test accuracy was then reported before the next selection round. The results below are *preliminary* results each obtained only with one run. We are doing a more thorough benchmarking experiment, with more runs and report standard deviations etc. We will also link to a preprint which will include the benchmarking results.

For more details on the benchmarking results, please check out the [Active Learning Benchmark Blog: Cut Down Labeling Costs with DISTIL](https://decile-research.medium.com/cut-down-on-labeling-costs-with-distil-77bec5c2e864).

#### CIFAR10
Model: Resnet18


![CIFAR10](./experiment_plots/cifar10_plot_50k.png?raw=true)


The best strategies show 2x labeling efficiency compared to random sampling. BADGE does better than entropy sampling with a larger budget, and all strategies do better than random sampling.


#### MNIST
Model: MnistNet


![MNIST Plot](./experiment_plots/mnist_plot.png?raw=true)

All strategies exhibit a gain over random sampling, and both entropy sampling and BADGE achieve a 4x labeling efficiency compared to random sampling.


#### FASHION MNIST
Model: Resnet18


![FMNIST Plot](./experiment_plots/fmnist_plot.png?raw=true)


All strategies exhibit a gain over random sampling, and both entropy sampling and BADGE achieve a 4x labeling efficiency compared to random sampling.


#### SVHN
Model: Resnet18


![SVHN Plot](./experiment_plots/svhn_plot.png?raw=true)


All strategies exhibit a gain over random sampling, and both entropy sampling and BADGE achieve a 3x labeling efficiency compared to random sampling.


#### OPENML-6
Budget: 400, Model: Two Layer Net, Number of rounds: 11, Total Points: 4800 (30%)

![OPENML6 Plot](./experiment_plots/openml6_plot.png?raw=true)

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

## Mailing List
To receive updates about DISTIL and to be a part of the community, join the Decile_DISTIL_Dev group.
```
https://groups.google.com/forum/#!forum/Decile_DISTIL_Dev/join 
```
## Acknowledgement
This library takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning), [Jordan Ash's Badge repository](https://github.com/JordanAsh/badge), and [Andreas Kirsch's and Joost van Amersfoort's BatchBALD repository](https://github.com/BlackHC/batchbald_redux). Also, DISTIL uses [Apricot](https://github.com/jmschrei/apricot) for submodular optimization.

## Team
DISTIL is created and maintained by Nathan Beck, [Durga Sivasubramanian](https://www.linkedin.com/in/durga-s-352831105), [Apurva Dani](https://apurvadani.github.io/index.html), [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have DISTIL more community driven. Please use it and contribute to it for your active learning research, and feel free to use it for your commercial projects. We will add the major contributors here.

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

