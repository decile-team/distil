Models
===========================

We have incorporated several neural network architectures in the DISTIL repository. Below given is a list of Neural network architectures:
 - densenet
 - dla
 - dla_simple
 - dpn
 - efficientnet
 - googlenet
 - lenet
 - mobilenet
 - mobilenetv2
 - pnasnet
 - preact_resnet
 - regnet
 - resnet
 - resnext
 - senet
 - shufflenet
 - shufflenetv2
 - vgg


**To use custom model architecture, modify the model architecture in the following way:**

The forward method should have two more variables:

#. A boolean variable *last* which -

	*If *true*: returns the model output and the output of the second last layer

	*If *false*: Returns the model output.

#. A boolean variable ‘freeze’ which -

	*If *true*: disables the tracking of any calculations required to later calculate a gradient i.e skips gradient calculation over the weights

	*If *false*: otherwise

#. get_embedding_dim() method which returns the number of hidden units in the last layer.
