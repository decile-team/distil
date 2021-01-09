# DISTIL: Deep dIverSified inTeractIve Learning
DISTIL implements a number of state of the art active learning algorithms. Some of the algorithms currently implemented with DISTIL include:

- 
- Uncertainty Sampling [1]
- Margin Sampling [2]
- Least Confidence Sampling [2]
- FASS [3]
- BADGE [2]
- GLISTER_ACTIVE [1]
- CoreSets based Active Learning [5]
- Ramdom Sampling
- Submodular Sampling [3,6,7]


Publications:

[1] Settles, Burr. Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009.

[2] Wang, Dan, and Yi Shang. "A new active labeling method for deep learning." 2014 International joint conference on neural networks (IJCNN). IEEE, 2014

[3] Kai Wei, Rishabh Iyer, Jeff Bilmes, Submodularity in data subset selection and active learning, International Conference on Machine Learning (ICML) 2015

[4] Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. CoRR, 2019. URL: http://arxiv.org/abs/1906.03671, arXiv:1906.03671.

[5] Sener, Ozan, and Silvio Savarese. "Active learning for convolutional neural networks: A core-set approach." ICLR 2018.

[6] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, 35th AAAI Conference on Artificial Intelligence, AAAI 2021 

[7] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[8] Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

## Installation
The latest version of  DISTIL package can be installed using the following command:

```python
pip install --extra-index-url https://test.pypi.org/simple/ decile-distil
```
---
**NOTE:**
  Please make sure to enter the space between simple/ and decile-distil in the above command while installing CORDS package
---

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "apricot-select >= 0.6.0"

## Demo Notebooks
1. https://colab.research.google.com/drive/10WkyKlOxSixrMHvA9wEHcd0l5HugnChN?usp=sharing

2. https://colab.research.google.com/drive/15427CIEy6rIDwfTWsprUH6yPfufjjY56?usp=sharing

3. https://colab.research.google.com/drive/1PaMne-hsAMlzZt6Aul3kZbOezx-2CgKc?usp=sharing
