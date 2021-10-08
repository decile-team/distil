#### Warm-Starting

AL practitioners may be tempted to use the working model in future AL rounds. To examine the effect of having this "warm start" model. More details can be found in [Effective Evaluation of Deep Active Learning on Image Classification Tasks](https://arxiv.org/abs/2106.15324).

![RESET](../../experiment_plots/persistence.png?raw=true)

Although the accuracies are close, we see that the labeling efficiency of each AL method is greater when using a model reset after each AL round. The effect is more pronounced when generalization techniques are not used:

![RESET2](../../experiment_plots/reset_update_appendix.png?raw=true)