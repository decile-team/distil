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

## What is DISTIL?
<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/distil/blob/main/experiment_plots/distil_explanation.png" width="543" height="390"/>
    </br>
</p>

DISTIL is an active learning toolkit that implements a number of state-of-the-art active learning strategies with a particular focus for active learning in the deep learning setting. DISTIL is built on *PyTorch* and decouples the training loop from the active learning algorithm, thereby providing flexibility to the user by allowing them to control the training procedure and model. It allows users to incorporate new active learning algorithms easily with minimal changes to their existing code. DISTIL also provides support for incorporating active learning with your custom dataset and allows you to experiment on well-known datasets. We are continuously incorporating newer and better active learning selection strategies into DISTIL, and we plan to expand the scope of the supported active learning algorithms to settings beyond the currently supported supervised classification setting.

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

DISTIL can be installed using the following means:

### From Git Repository
```
git clone https://github.com/decile-team/distil.git
cd distil
pip install -r requirements/requirements.txt
```

### Pip Installation
```python
pip install decile-distil
```

### First Steps

To better understand DISTIL's functionality, we have provided example Jupyter notebooks in the [tutorials](tutorials) folder, which can be easily executed by using Google Colab. We also provide a simple AL training loop that runs experiments using a provided configuration file. To run this loop, do the following from the base folder:

```python
python train.py --config_path=/content/distil/configs/config_svhn_resnet_randomsampling.json
```

You can use the default configurations that we have provided in the configs folder, or you can make a custom configuration. For making your custom configuration file for training, please refer to [Distil Configuration File Documentation](https://decile-team-distil.readthedocs.io/en/latest/configuration.html).

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

## Documentation
Learn more about DISTIL by reading our [documentation](https://decile-team-distil.readthedocs.io/en/latest/).

## Mailing List
To receive updates about DISTIL and to be a part of the community, join the Decile_DISTIL_Dev group.
```
https://groups.google.com/forum/#!forum/Decile_DISTIL_Dev/join 
```
## Acknowledgment
This library takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning), [Jordan Ash's Badge repository](https://github.com/JordanAsh/badge), and [Andreas Kirsch's and Joost van Amersfoort's BatchBALD repository](https://github.com/BlackHC/batchbald_redux). Also, DISTIL uses [submodlib](https://github.com/decile-team/submodlib) for submodular optimization.

## Team
DISTIL is created and maintained by Nathan Beck, [Suraj Kothawade](https://personal.utdallas.edu/~snk170001/), [Durga Sivasubramanian](https://www.linkedin.com/in/durga-s-352831105), [Apurva Dani](https://apurvadani.github.io/index.html), [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have DISTIL more community driven. Please use it and contribute to it for your active learning research, and feel free to use it for your commercial projects. We will add the major contributors here.

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

[7] Vishal Kaushal, Rishabh Iyer, Suraj Kothawade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[8] Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

[9] Ducoffe, Melanie, and Frederic Precioso. "Adversarial active learning for deep networks: a margin based approach." arXiv preprint arXiv:1802.09841 (2018).

[10] Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian active learning with image data." International Conference on Machine Learning. PMLR, 2017.

[11] Suraj Kothawade, Nathan Beck, Krishnateja Killamsetty, and Rishabh Iyer, “SIMILAR: Submodular Information Measures based Active Learning in Realistic Scenarios,”
To Appear In Neural Information Processing Systems, NeurIPS 2021.

