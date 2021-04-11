Configuration Files for Training
================================

This page gives a tutorial on how to generate your custom training configuration files.

This configuration files can be used to select datasets, training configuration, and active learning settings. These files are in json format.

.. code-block:: json

	{
		"model": {
			"architecture": "resnet18",
			"target_classes": 10
		},
		"train_parameters": {
			"lr": 0.001,
			"batch_size": 1000,
			"n_epoch": 50,
			"max_accuracy": 0.95,
			"isreset": true,
			"islogs":  true,
			"logs_location": "./logs.txt"
		},

		"active_learning":{
			"strategy": "badge",
			"budget": 1000,
			"rounds": 15,
			"initial_points":1000,
			
			"strategy_args":{	
				"batch_size" : 1000, 
				"lr":0.001
			}
		},
		"dataset":{
			"name":"cifar10"
		}
	}


The configuration files consists of following sections:

#. Model
#. Training Parameters
#. Active Learning Configuration
#. Dataset

**Symbol (%) represents mandatory arguments**

**model**

#. architecture % 
	* Model architecture to be used, Presently it supports the below mentioned architectures.
		#. resnet18
		#. two_layer_net
#. target_classes %
	* Number of output classes for prediction. 
#. input_dim
	* Input dimension of the dataset. To be mentioned while using two layer net.
#. hidden_units_1
	* Number of hidden units to be used in the first layer. To be mentioned while using two layer net.

**train_parameters**

#. lr %
	* Learning rate to be used for training.
#. batch_size %
	* Batch size to be used for training.
#. n_epoch %
	* Maximum number of epochs for the model to train.
#. max_accuracy
	* Maximum training accuracy after which training should be stopped.
#. isreset
	* Reset weight whenever the model training starts.
		#. True
		#. False
#. islogs
	* Log training output.
		#. True
		#. False
#. logs_location %
	* Location where logs should be saved.

**active_learning**

#. strategy %
	* Active learning strategy to be used.
		#. badge
		#. glister
		#. entropy_sampling
		#. margin_sampling
		#. least_confidence
		#. core_set
		#. random_sampling
		#. fass
		#. bald_dropout
		#. adversarial_bim
		#. kmeans_sampling
		#. baseline_sampling
		#. adversarial_deepfool
#. budget %
	* Number of points to be selected by the active learning strategy.
#. rounds %
	* Total number of rounds to run active learning for.
#. initial_points
	* Initial number of points to start training with.
#. strategy_args
	* Arguments to pass to the strategy. It varies from strategy to strategy. Please refer to the documentation of the strategy that is being used.

**dataset**

#. name
	* Name of the dataset to be used. It presently supports following datasets.
		#. cifar10
		#. mnist
		#. fmnist
		#. svhn
		#. cifar100
		#. satimage
		#. ijcnn1

You can refer to various configuration examples in the configs/ folders of the DISTIL repository.