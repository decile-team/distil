import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def dict_to(dictionary, device):
    
    # Predict the most likely class
    if type(dictionary) == dict:
        for key in dictionary:
            value = dictionary[key]
            if hasattr(value, "to"):
                dictionary[key] = value.to(device=device)
    
    return dictionary
                        

class Strategy:
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.model = net
        self.target_classes = nclasses
        self.args = args
        
        if 'batch_size' not in args:
            args['batch_size'] = 1
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']
            
        if 'loss' not in args:
            self.loss = F.cross_entropy
        else:
            self.loss = args['loss']

    def select(self, budget):
        pass

    def update_data(self, labeled_dataset, unlabeled_dataset): #
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        
    def update_queries(self, query_dataset):
        self.query_dataset= query_dataset

    def update_privates(self, private_dataset):
        self.private_dataset= private_dataset

    def update_model(self, clf):
        self.model = clf

    def predict(self, to_predict_dataset):
    
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold class predictions
        P = torch.zeros(len(to_predict_dataset)).long().to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():
            for elements_to_predict in to_predict_dataloader:
                
                # Predict the most likely class
                if type(elements_to_predict) == dict:
                    elements_to_predict = dict_to(elements_to_predict, self.device)
                    out = self.model(**elements_to_predict)
                else:
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                pred = out.max(1)[1]
                
                # Insert the calculated batch of predictions into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + pred.shape[0]
                P[start_slice:end_slice] = pred
                evaluated_instances = end_slice
                
        return P

    def predict_prob(self, to_predict_dataset):

        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():
            for elements_to_predict in to_predict_dataloader:
                
                # Calculate softmax (probabilities) of predictions
                if type(elements_to_predict) == dict:
                    elements_to_predict = dict_to(elements_to_predict, self.device)
                    out = self.model(**elements_to_predict)
                else:
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                pred = F.softmax(out, dim=1)
                
                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + pred.shape[0]
                probs[start_slice:end_slice] = pred
                evaluated_instances = end_slice

        return probs        

    def predict_prob_dropout(self, to_predict_dataset, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for elements_to_predict in to_predict_dataloader:
                
                    # Calculate softmax (probabilities) of predictions
                    if type(elements_to_predict) == dict:
                        elements_to_predict = dict_to(elements_to_predict, self.device)
                        out = self.model(**elements_to_predict)
                    else:
                        elements_to_predict = elements_to_predict.to(self.device)
                        out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + pred.shape[0]
                    probs[start_slice:end_slice] += pred
                    evaluated_instances = end_slice

        # Divide through by n_drop to get average prob.
        probs /= n_drop

        return probs         

    def predict_prob_dropout_split(self, to_predict_dataset, n_drop):
        
        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataset), self.target_classes]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                evaluated_instances = 0
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    if type(elements_to_predict) == dict:
                        elements_to_predict = dict_to(elements_to_predict, self.device)
                        out = self.model(**elements_to_predict)
                    else:
                        elements_to_predict = elements_to_predict.to(self.device)
                        out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + pred.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs 

    def get_embedding(self, to_predict_dataset):
        
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold embeddings
        embedding = torch.zeros([len(to_predict_dataset), self.model.get_embedding_dim()]).to(self.device)
        
        # Create a dataloader object to load the dataset
        to_predict_dataloader = DataLoader(to_predict_dataset, batch_size = self.args['batch_size'], shuffle = False)
        
        evaluated_instances = 0
        
        with torch.no_grad():

            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Calculate softmax (probabilities) of predictions
                if type(elements_to_predict) == dict:
                    elements_to_predict = dict_to(elements_to_predict, self.device)
                    out, l1 = self.model(**elements_to_predict, last=True)
                else:    
                    elements_to_predict = elements_to_predict.to(self.device)
                    out, l1 = self.model(elements_to_predict, last=True)
                
                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + out.shape[0]
                embedding[start_slice:end_slice] = l1
                evaluated_instances = end_slice

        return embedding

    def _last_layer_backprop(self, model_output, l1, targets, grad_embedding_type):
    
        embDim = self.model.get_embedding_dim()
        
        # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
        loss = self.loss(model_output, targets, reduction="sum")
        l0_grads = torch.autograd.grad(loss, model_output)[0]

        # Calculate the linear layer gradients as well if needed
        if grad_embedding_type != "bias":
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.target_classes)

        # Populate embedding tensor according to the supplied argument.
        if grad_embedding_type == "bias":                
            gradient_embedding = l0_grads
        elif grad_embedding_type == "linear":
            gradient_embedding = l1_grads
        else:
            gradient_embedding = torch.cat([l0_grads, l1_grads], dim=1) 
            
        return gradient_embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self, dataset, predict_labels, grad_embedding_type="bias_linear"):
        
        embDim = self.model.get_embedding_dim()
        self.model = self.model.to(self.device)
        
        # Create the tensor to return depending on the grad_embedding_type, which can have bias only, 
        # linear only, or bias and linear
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(dataset), self.target_classes]).to(self.device)
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(dataset), embDim * self.target_classes]).to(self.device)
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros([len(dataset), (embDim + 1) * self.target_classes]).to(self.device)
        else:
            raise ValueError("Grad embedding type not supported: Pick one of 'bias', 'linear', or 'bias_linear'")
          
        # Create a dataloader object to load the dataset
        dataloader = DataLoader(dataset, batch_size = self.args['batch_size'], shuffle = False)  
          
        evaluated_instances = 0
        
        # We need to be careful how we unpack each batch from the loader. Each case
        # depends on if 1) the dataloader returns dictionary objects and 2) we need
        # to predict labels. Each is enumerated here. DataLoader will return dict
        # objects if dataset returns dictionaries.
        dataloader_is_returning_dictionary_type_object = type(dataset[0]) == dict
        
        if dataloader_is_returning_dictionary_type_object:
            
            for data_batch_dict in dataloader:
                start_slice = evaluated_instances
                
                data_batch_dict = dict_to(data_batch_dict, self.device)
                
                if not predict_labels:
                    targets = data_batch_dict["labels"] # We expect labels to be in "labels" field of dictionary
                    del data_batch_dict["labels"]
                
                out, l1 = self.model(**data_batch_dict, last=True, freeze=True)
            
                if predict_labels:
                    targets = out.max(1)[1]
                    
                end_slice = start_slice + targets.shape[0]

                grad_embedding[start_slice:end_slice] = self._last_layer_backprop(out, l1, targets, grad_embedding_type)
            
                evaluated_instances = end_slice
            
                # Empty the cache as the gradient embeddings could be very large
                torch.cuda.empty_cache()
        else:
            
            if predict_labels:
                for unlabeled_data_batch in dataloader:
                    start_slice = evaluated_instances
                    
                    inputs = unlabeled_data_batch.to(self.device, non_blocking=True)
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    targets = out.max(1)[1]
                        
                    end_slice = start_slice + targets.shape[0]
        
                    grad_embedding[start_slice:end_slice] = self._last_layer_backprop(out, l1, targets, grad_embedding_type)
        
                    evaluated_instances = end_slice
                
                    # Empty the cache as the gradient embeddings could be very large
                    torch.cuda.empty_cache()
            else:
                for inputs, targets in dataloader:
                    start_slice = evaluated_instances
                    
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    out, l1 = self.model(inputs, last=True, freeze=True)
                        
                    end_slice = start_slice + targets.shape[0]
                
                    grad_embedding[start_slice:end_slice] = self._last_layer_backprop(out, l1, targets, grad_embedding_type)
                
                    evaluated_instances = end_slice
                
                    # Empty the cache as the gradient embeddings could be very large
                    torch.cuda.empty_cache()
        
        # Return final gradient embedding
        return grad_embedding

    def feature_extraction(self, inp, layer_name):
        feature = {}
        model = self.model
        def get_features(name):
            def hook(model, inp, output):
                feature[name] = output.detach()
            return hook
        for name, layer in self.model._modules.items():
            if name == layer_name:
                layer.register_forward_hook(get_features(layer_name))
        
        with torch.no_grad():
            if type(inp) == dict:
                output = self.model(**inp)
            else:
                output = self.model(inp)
            
        return torch.squeeze(feature[layer_name])

    def get_feature_embedding(self, dataset, unlabeled, layer_name='avgpool'):
        dataloader = DataLoader(dataset, batch_size = self.args['batch_size'], shuffle = False)
        features = []
        self.model = self.model.to(self.device)
        
        # Again, we need to be careful when unpacking instances from the loader. 
        # DataLoader will return dict objects if dataset returns dictionaries.
        dataloader_is_returning_dictionary_type_object = type(dataset[0]) == dict
        
        if dataloader_is_returning_dictionary_type_object:
            for inputs_dict in dataloader:
                inputs_dict = dict_to(inputs_dict, self.device)
                batch_features = self.feature_extraction(inputs_dict, layer_name)
                features.append(batch_features)
        else:
            if unlabeled:
                for inputs in dataloader:
                    inputs = inputs.to(self.device)
                    batch_features = self.feature_extraction(inputs, layer_name)
                    features.append(batch_features)
            else:
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    batch_features = self.feature_extraction(inputs, layer_name)
                    features.append(batch_features)
        
        return torch.vstack(features)