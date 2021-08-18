import numpy as np
import torch
from .strategy import Strategy
from torch.autograd import Variable
import collections
import copy

# Reflects the most recent version of zero_gradients before it 
# was removed from PyTorch's current deployment.
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

class AdversarialDeepFool(Strategy):
    """
    Implements Adversial Deep Fool Strategy :footcite:`ducoffe2018adversarial`, a Deep-Fool based 
    Active Learning strategy that selects unlabeled samples with the smallest adversarial 
    perturbation. This technique is motivated by the fact that often the distance computation 
    from decision boundary is difficult and intractable for margin-based methods. This 
    technique avoids estimating distance by using Deep-Fool :footcite:`Moosavi-Dezfooli_2016_CVPR` 
    like techniques to estimate how much adversarial perturbation is required to cross the boundary. 
    The smaller the required perturbation, the closer the point is to the boundary.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **max_iter**: Maximum Number of Iterations (int, optional)
    """
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        """
        Constructor method
        """
        if 'max_iter' in args:
            self.max_iter = args['max_iter']
        else:
            self.max_iter = 50
            
        super(AdversarialDeepFool, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args={})


    def deepfool(self, image, net, num_classes=10, overshoot=0.02):

        image = image.to(self.device)
        net = net.to(self.device)

        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).flatten()
        I = f_image.argsort(descending=True)

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.shape
        pert_image = image.clone()
        w = torch.zeros(input_shape).to(self.device)
        r_tot = torch.zeros(input_shape).to(self.device)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x)
        fs_list = [fs[0,I[k]] for k in range(num_classes)]
        k_i = label

        while k_i == label and loop_i < self.max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.clone()

            for k in range(1, num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.clone()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data

                pert_k = abs(f_k)/torch.sqrt(torch.dot(w_k.flatten(), w_k.flatten())).item()

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / torch.sqrt(torch.dot(w.flatten(),w.flatten()))
            r_tot = r_tot + r_i

            pert_image = image + (1+overshoot)*r_tot.to(self.device)

            x = Variable(pert_image, requires_grad=True)
            fs = net.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1+overshoot)*r_tot

        return torch.dot(r_tot.flatten(), r_tot.flatten())

    def select(self, budget):
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        self.model.eval()
        self.model = self.model.to(self.device)
        
        dis = np.zeros(len(self.unlabeled_dataset))
        data_pool = self.unlabeled_dataset
        for i in range(len(self.unlabeled_dataset)):
            x = data_pool[i]
            dist = self.deepfool(x, self.model, self.target_classes)
            dis[i] = dist
        
        idxs = dis.argsort()[:budget]
        return idxs