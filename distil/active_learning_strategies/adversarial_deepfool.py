import numpy as np
import torch
from .strategy import Strategy
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy
from torch.autograd.gradcheck import zero_gradients

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
    X: numpy array
        Present training/labeled data   
    y: numpy array
        Labels of present training data
    unlabeled_x: numpy array
        Data without labels
    net: class
        Pytorch Model class
    handler: class
        Data Handler, which can load data even without labels.
    nclasses: int
        Number of unique target variables
    args: dict
        Specify optional parameters
        
        batch_size 
        Batch size to be used inside strategy class (int, optional)

        max_iter
        Maximum Number of Iterations (int, optional)
    """
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        """
        Constructor method
        """
        if 'max_iter' in args:
            self.max_iter = args['max_iter']
        else:
            self.max_iter = 50        
        super(AdversarialDeepFool, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})


    def deepfool(self, image, net, num_classes=10, overshoot=0.02, max_iter=50):

        """
        :param image: Image of size HxWx3
        :param net: network (input: images, output: values of activation **BEFORE** softmax).
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for deepfool (default = 50)
        :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
        """

        image = image.to(self.device)
        net = net.to(self.device)

        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).flatten()
        I = f_image.argsort(descending=True)

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.shape
        pert_image = copy.deepcopy(image)
        w = torch.zeros(input_shape).to(self.device)
        r_tot = torch.zeros(input_shape).to(self.device)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x)
        fs_list = [fs[0,I[k]] for k in range(num_classes)]
        k_i = label

        while k_i == label and loop_i < max_iter:

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
        

    def cal_dis(self, x):

        nx = Variable(torch.unsqueeze(x, 0), requires_grad=True)
        eta = Variable(torch.zeros(nx.shape))

        out = self.model(nx + eta)
        n_class = out.shape[1]
        py = int(out.max(1)[1])
        ny = int(out.max(1)[1])

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(float(fi)) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += Variable(ri.clone())
            nx.grad.data.zero_()
            out = self.model(nx + eta)
            py = int(out.max(1)[1])
            i_iter += 1

        return (eta*eta).sum()

    def select(self, budget):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set

        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        self.model.eval()
        dis = np.zeros(self.unlabeled_x.shape[0])
        data_pool = self.handler(self.unlabeled_x)
        for i in range(self.unlabeled_x.shape[0]):
            x, idx = data_pool[i]
            dist = self.deepfool(x, self.model, self.target_classes)
            dis[i] = dist

        self.model.to(self.device)
        idxs = dis.argsort()[:budget]
        return idxs


