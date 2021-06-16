import torch
import torch.nn.functional as F

class Strategy:
    def __init__(self, labeled_dataloader, unlabeled_dataloader, net, nclasses, args={}): #
        
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
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

    def update_data(self, labeled_dataloader, unlabeled_dataloader): #
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader

    def update_model(self, clf):
        self.model = clf

    def predict(self, to_predict_dataloader):
    
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold class predictions
        P = torch.zeros(len(to_predict_dataloader.dataset)).long().to(self.device)
        
        with torch.no_grad():
            # Instantiate the lower slice index to insert this batch's predictions into P
            low_insert_index = 0
            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Predict the most likely class
                elements_to_predict = elements_to_predict.to(self.device)
                out = self.model(elements_to_predict)
                pred = out.max(1)[1]
                
                # Calculate high slice index
                current_batch_size = elements_to_predict.shape[0]
                high_insert_index = low_insert_index + current_batch_size
                
                # Insert the calculated batch of predictions into the tensor to return
                P[low_insert_index:high_insert_index] = pred
                # Update lower slice index for next batch
                low_insert_index = high_insert_index

        return P

    def predict_prob(self, to_predict_dataloader):

        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataloader.dataset), self.target_classes]).to(self.device)
        
        with torch.no_grad():
            # Instantiate the lower slice index to insert this batch's predictions into probs
            low_insert_index = 0
            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict.to(self.device)
                out = self.model(elements_to_predict)
                pred = F.softmax(out, dim=1)
                
                # Calculate high slice index
                current_batch_size = elements_to_predict.shape[0]
                high_insert_index = low_insert_index + current_batch_size
                
                # Insert the calculated batch of probabilities into the tensor to return
                probs[low_insert_index:high_insert_index] = pred
                
                # Update lower slice index for next batch
                low_insert_index = high_insert_index

        return probs        

    def predict_prob_dropout(self, to_predict_dataloader, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_dataloader.dataset), self.target_classes]).to(self.device)
        
        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                # Instantiate the lower slice index to insert this batch's predictions into probs
                low_insert_index = 0
                
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Calculate high slice index
                    current_batch_size = elements_to_predict.shape[0]
                    high_insert_index = low_insert_index + current_batch_size
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    probs[low_insert_index:high_insert_index] += pred
                
                    # Update lower slice index for next batch
                    low_insert_index = high_insert_index

        # Divide through by n_drop to get average prob.
        probs /= n_drop

        return probs         

    def predict_prob_dropout_split(self, to_predict_dataloader, n_drop):
        
        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.model.train()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_dataloader.dataset), self.target_classes]).to(self.device)
        
        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):
                
                # Instantiate the lower slice index to insert this batch's predictions into P
                low_insert_index = 0
                
                for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict.to(self.device)
                    out = self.model(elements_to_predict)
                    pred = F.softmax(out, dim=1)
                
                    # Calculate high slice index
                    current_batch_size = elements_to_predict.shape[0]
                    high_insert_index = low_insert_index + current_batch_size
                
                    # Accumulate the calculated batch of probabilities into the tensor to return
                    probs[i][low_insert_index:high_insert_index] = pred
                
                    # Update lower slice index for next batch
                    low_insert_index = high_insert_index

        return probs 

    def get_embedding(self, to_predict_dataloader):
        
        # Ensure model is on right device and is in eval. mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Create a tensor to hold embeddings
        embedding = torch.zeros([len(to_predict_dataloader.dataset), self.model.get_embedding_dim()]).to(self.device)
        
        with torch.no_grad():
            # Instantiate the lower slice index to insert this batch's embeddings into embedding
            low_insert_index = 0
            for batch_idx, elements_to_predict in enumerate(to_predict_dataloader):
                
                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict.to(self.device)
                out, l1 = self.model(elements_to_predict, last=True)
                
                # Calculate high slice index
                current_batch_size = elements_to_predict.shape[0]
                high_insert_index = low_insert_index + current_batch_size
                
                # Insert the calculated batch of probabilities into the tensor to return
                embedding[low_insert_index:high_insert_index] = l1
                
                # Update lower slice index for next batch
                low_insert_index = high_insert_index

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self, dataloader, predict_labels, grad_embedding_type="bias_linear"):
        
        embDim = self.model.get_embedding_dim()
        self.model = self.model.to(self.device)
        
        # Create the tensor to return depending on the grad_embedding_type, which can have bias only, 
        # linear only, or bias and linear
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(dataloader.dataset), self.target_classes]).to(self.device)
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(dataloader.dataset), embDim]).to(self.device)
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros([len(dataloader.dataset), (embDim + 1) * self.target_classes]).to(self.device)
        else:
            raise ValueError("Grad embedding type not supported: Pick one of 'bias', 'linear', or 'bias_linear'")
        
        low_insert_index = 0
        
        for batch_idx, possible_data_label_pair in enumerate(dataloader):
            
            # If labels need to be predicted, then do so. Calculate output as normal.
            if predict_labels:
                inputs = possible_data_label_pair.to(self.device, non_blocking=True)
                out, l1 = self.model(inputs, last=True, freeze=True)
                targets = out.max(1)[1]
            else:
                inputs, targets = possible_data_label_pair[0].to(self.device, non_blocking=True), possible_data_label_pair[1].to(self.device, non_blocking=True)
                out, l1 = self.model(input, last=True, freeze=True)
            
            # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
            loss = self.loss(out, targets, reduction="sum")
            l0_grads = torch.autograd.grad(loss, out)[0]

            # Calculate the linear layer gradients as well if needed
            if grad_embedding_type != "bias":
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * l1.repeat(1, self.target_classes)

            # Calculate high slice index
            current_batch_size = inputs.shape[0]
            high_insert_index = low_insert_index + current_batch_size
                        
            # Populate embedding tensor according to the supplied argument.
            if grad_embedding_type == "bias":                
                grad_embedding[low_insert_index:high_insert_index] = l0_grads
            elif grad_embedding_type == "linear":
                grad_embedding[low_insert_index:high_insert_index] = l1_grads
            else:
                grad_embedding[low_insert_index:high_insert_index] = torch.cat([l0_grads, l1_grads], dim=1) 
            
            # Empty the cache as the gradient embeddings could be very large
            torch.cuda.empty_cache()
            
            # Update low insert index
            low_insert_index = high_insert_index
        
        # Return final gradient embedding
        return grad_embedding