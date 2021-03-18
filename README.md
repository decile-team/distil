<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/distil/blob/main/experiment_plots/distil_logo_transparent.png" width="500" height="150"/>
    </br>
    <br>
        <strong> Deep dIverSified inTeractIve Learning </strong>
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
<p>Label only the required data and reduce labeling cost and time.
</h3>


# In this README
- [What is DISTIL?](#what-is-distil)
- [Where can DISTIL be used?](#where-can-distil-be-used)
- [Installation](#installation)
- [Package Requirements](#package-requirements)
- [Documentation](#documentation)
- [Demo Notebooks](#demo-notebooks)
- [Evaluation of Active Learning Strategies](#evaluation-of-active-learning-strategies)
- [Testing Individual strategy](#testing-individual-strategy)
- [Mailing List](#mailing-list)
- [Publications](#publications)
- [Acknowledgement](#acknowledgement)

## What is DISTIL?
DISTIL implements a number of state of the art active learning algorithms. The following are the goals that DISTIL tries to acheive:
<p align="center"><i><b>Reducing Labeling Time</b></i></p>
<p align="center"><i><b>Reducing Labeling Man Power & Costs</b></i></p>
<p align="center"><i><b>Reducing Amount of Data to be Labeled</b></i></p>
<p align="center"><i><b>Reducing End to End Training Time</b></i></p>
<p align="center"><i><b>Reducing Energy Requirement</b></i></p>
<p align="center"><i><b>Reducing Resource (GPU) Requirement and Costs</b></i></p>

Some of the algorithms currently implemented with DISTIL include:

- [Uncertainty Sampling [1]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.entropy_sampling)
- [Margin Sampling [2]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.margin_sampling)
- [Least Confidence Sampling [2]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.least_confidence)
- [FASS [3]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.fass)
- [BADGE [4]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.badge)
- [GLISTER ACTIVE [6]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.glister)
- [CoreSets based Active Learning [5]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.core_set)
- [Ramdom Sampling](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.random_sampling)
- [Submodular Sampling [3,6,7]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.submod_sampling)
- [Adversarial DeepFool [9]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.adversarial_deepfool)
- [BALD [10]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.bayesian_active_learning_disagreement_dropout)
- [Kmeans Sampling [5]](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.kmeans_sampling)
- [Adversarial Bim](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.adversarial_bim)
- [Baseline Sampling](https://decile-team-distil.readthedocs.io/en/latest/ActStrategy/distil.active_learning_strategies.html#module-distil.active_learning_strategies.baseline_sampling)

We are continuously incorporating newer and better selection strategies into DISTIL. Some of the features of DISTIL includes:
- Minimal changes to add it to the existing training structure.
- Independent of the training strategy used.
- Achieving similar test accuracy with less amount of training data.
- Huge reduction in labelling cost and time.
- Access to various active learning strategies with just one line of code.

## Where can DISTIL be used?
DISTIL is a toolkit which provides support for various active learning algorithms. Presently it only works with classification task. It can be used in scenarios where you only want to label few data points which can provide maximum information to the classification model and thus reduce labeling cost and time.

## Installation
The latest version of  DISTIL package can be installed using the following command:

```python
pip install --extra-index-url https://test.pypi.org/simple/ decile-distil
```
### NOTE
```
Please make sure to enter the space between simple/ and decile-distil in the above command while installing DISTIL package
```

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "apricot-select >= 0.6.0"

## Documentation
Learn more about distil at our [documentation](https://decile-team-distil.readthedocs.io/en/latest/).

## Demo Notebooks
1. https://colab.research.google.com/drive/10WkyKlOxSixrMHvA9wEHcd0l5HugnChN?usp=sharing

2. https://colab.research.google.com/drive/15427CIEy6rIDwfTWsprUH6yPfufjjY56?usp=sharing

3. https://colab.research.google.com/drive/1PaMne-hsAMlzZt6Aul3kZbOezx-2CgKc?usp=sharing

## Evaluation of Active Learning Strategies
### Experimentation Method
The model was first trained on randomly selected n points where n is the budget of the experiment. For each set of new points added, the model was trained from scratch till the training accuracy crossed 95%.

### CIFAR10
Budget: 1000, Model: Resnet18, Number of rounds: 14, Total Points: 15,000 (30%)

![CIFAR10 Plot](./experiment_plots/cifar10_plot.png?raw=true)

| Strategy | Accuracy |
| --- | --- |
| BADGE | 0.699 | 
| FASS | 0.691 | 
| Entropy Sampling | 0.688 | 
| Glister | 0.676 | 
| Random Sampling | 0.648 | 
| Margin Sampling | 0.638 | 
| Coreset | 0.632 | 

### MNIST
Budget: 1000, Model: Resnet18, Number of rounds: 11, Total Points: 12,000 (20%)

![MNIST Plot](./experiment_plots/mnist_plot.png?raw=true)

| Strategy | Accuracy |
| --- | --- |
| Margin Sampling |	0.993 |
| Entropy Sampling | 0.993 |
| BADGE | 0.992 | 
| Glister |	0.992 |
| Coreset |	0.991 |
| FASS |	0.99 |
| Random Sampling |	0.98 |

### OPENML-6
Budget: 400, Model: Two Layer Net, Number of rounds: 11, Total Points: 4800 (30%)

![OPENML6 Plot](./experiment_plots/openml6_plot.png?raw=true)

| Strategy | Accuracy |
| --- | --- |
| Glister | 0.938 |
| BADGE | 0.938 |
| FASS | 0.935 |
| Margin Sampling |	0.934 |
| Coreset |	0.925 |
| Entropy Sampling | 0.924 |
| Random Sampling |	0.911 |

## Testing Individual strategy
If there are any changes made in the strategy, it can be tested using distil.utils.TestStrategy
```
from distil.utils.TestStrategy import test_strategy
test_strategy('badge')
```
## Mailing List
To receive updates about distil and be a part of the community, join the Decile_DISTIL_Dev group.
```
https://groups.google.com/forum/#!forum/Decile_DISTIL_Dev/join 
```

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

## Acknowledgement
This library takes inspiration and also uses pieces of code from [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning) and [JordanAsh's Badge repository](https://github.com/JordanAsh/badge).
