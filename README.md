# DISTIL: Deep dIverSified inTeractIve Learning
DISTIL implements a number of state of the art active learning algorithms. Some of the algorithms currently implemented with DISTIL include:

- GLISTER_ACTIVE [1]
- FASS[3]
- BADGE [2]
- SubmodularSelection [4,5]
  - Facility Location
  - Feature Based Functions
  - Coverage
  - Diversity
- RandomSelection

Publications:

[1] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[2] Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. CoRR, 2019. URL: http://arxiv.org/abs/1906.03671, arXiv:1906.03671.

[3] Kai Wei, Rishabh Iyer, Jeff Bilmes, Submodularity in data subset selection and active learning, International Conference on Machine Learning (ICML) 2015: 

[4] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[5] Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

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
