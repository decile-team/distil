## Active Learning Benchmarking using DISTIL
We include a thorough benchmarking of various AL algorithms that covers many evaluation facets. Our experiments can be found in the benchmark_notebooks folder. To execute these experiments, upload a selected experiment to Google Colab and connect to a hosted runtime. We present the results of this benchmark in each subfolder. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

## Folder Descriptions

| Folder | Description |
| :--- | :----: |
| [augmentation](augmentation) | Contains notebooks that examine the effect that data augmentation has on the evolution of test accuracy and labeling efficiency in AL |
| [baseline](baseline) | Contains notebooks that examine the evolution of test accuracy and labeling efficiency of many AL algorithms in baseline settings |
| [budget](budget) | Contains notebooks that examine the effect that the AL budget has on the evolution of test accuracy in AL |
| [ex_per_class](ex_per_class) | Contains notebooks that examine the effect of the number of unlabeled examples per class on the evolution of test accuracy and labeling efficiency in AL |
| [gradmatch](gradmatch) | Contains a notebook that examines the effect of using DSS techniques for faster training on the evolution of test accuracy in AL |
| [optimizer](optimizer) | Contains notebooks that examine the effect of using adaptive optimizers such as [Adam](https://arxiv.org/abs/1412.6980) on the evolution of test accuracy and labeling efficiency in AL |
| [redundancy](redundancy) | Contains notebooks that examine the effect of redundancy in the unlabeled set on the evolution of test accuracy and labeling efficiency across a few algorithms in AL |
| [seed](seed) | Contains notebooks that examine the effect of the type of seed set used (randomly selected v. more carefully constructed) on the evolution of test accuracy in AL |
| [swa_ss](swa_ss) | Contains notebooks that examine the effect of the use of generalization techniques in deep learning on the evolution of test accuracy and labeling efficiency across a few algorithms in AL |
| [warm_start](warm_start) | Contains notebooks that examine the effect of maintaining previously learned models as warm-starts for the next AL round on the evolution of test accuracy in AL |