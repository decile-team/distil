import math
import numpy as np
import torch
from torchvision import datasets

def add_label_noise(y_trn, num_cls, noise_ratio=0.8):
    """
    Adds noise to the specified list of labels.
    This functionality is taken from CORDS and 
    applied here.

    Parameters
    ----------
    y_trn : list
        The list of labels to add noise.
    num_cls : int
        The number of classes possible in the list.
    noise_ratio : float, optional
        The percentage of labels to modify. The default is 0.8.

    Returns
    -------
    y_trn : list
        The list of now-noisy labels

    """
    noise_size = int(len(y_trn) * noise_ratio)
    noise_indices = np.random.choice(np.arange(len(y_trn)), size=noise_size, replace=False)
    y_trn[noise_indices] = np.random.choice(np.arange(num_cls), size=noise_size, replace=True)
    return y_trn

def get_imbalanced_idx(y_trn, num_cls, class_ratio=0.6):
    """
    Returns a list of indices of the supplied dataset that 
    constitute a class-imbalanced subset of the supplied 
    dataset. This functionality is taken from CORDS and 
    applied here.

    Parameters
    ----------
    y_trn : numpy ndarray
        The label set to choose imbalance.
    num_cls : int
        The number of classes possible in the list.
    class_ratio : float, optional
        The percentage of classes to affect. The default is 0.6.

    Returns
    -------
    subset_idxs : list
        The list of indices of the supplied dataset that 
        constitute a class-imbalanced subset
    """
    
    # Calculate the minimum samples in a class and take a small fraction of that number as the new sample 
    # count for that class
    samples_per_class = torch.zeros(num_cls)
    for i in range(num_cls):
        samples_per_class[i] = len(torch.where(torch.Tensor(y_trn) == i)[0])
    min_samples = int(torch.min(samples_per_class) * 0.1)
    
    # Generate affected classes based on the specified class ratio
    selected_classes = np.random.choice(np.arange(num_cls), size=int(class_ratio * num_cls), replace=False)
    
    # For each class, either add the full class to the dataset (if not selected) or add only min_samples
    # samples from that class to the dataset
    for i in range(num_cls):
        if i == 0:
            if i in selected_classes:
                subset_idxs = list(
                    np.random.choice(torch.where(torch.Tensor(y_trn) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
            else:
                subset_idxs = list(torch.where(torch.Tensor(y_trn) == i)[0].cpu().numpy())
        else:
            if i in selected_classes:
                batch_subset_idxs = list(
                    np.random.choice(torch.where(torch.Tensor(y_trn) == i)[0].cpu().numpy(),
                                             size=min_samples,
                                             replace=False))
            else:
                batch_subset_idxs = list(torch.where(torch.Tensor(y_trn) == i)[0].cpu().numpy())
            subset_idxs.extend(batch_subset_idxs)

    return subset_idxs

def make_data_redundant(X,Y,intial_bud,unique_points= 5000,amtRed=2):

    """
    Modifies the input dataset in such a way that only X.shape(0)/amtRed are original 
    points and rest are repeated or redundant. 

    Parameters
    ----------
    X : numpy ndarray
        The feature set to be made redundant.
    Y : numpy ndarray
        The label set corresponding to the X.
    intial_bud : int
        Number of inital points that are assumed to be labled. 
    unique_points: int
        Number of points to be kept unique in unlabled pool. 
    amtRed : float, optional
        Factor that determines redundancy. The default is 2.

    Returns
    -------
    X : numpy ndarray
        Modified feature set.
    """

    unique_ind = intial_bud + unique_points
    
    classes,no_elements = np.unique(Y[unique_ind:], return_counts=True)

    for cl in range(len(classes)):
        retain = math.ceil(no_elements[cl]/amtRed)
        idxs = np.where(Y[unique_ind:] == classes[cl])[0]

        idxs += unique_ind

        for i in range(math.ceil(amtRed)):
            if i == 0:
                idxs_rep = idxs[:retain]
            else: 
                idxs_rep = np.concatenate((idxs_rep,idxs[:retain]),axis=0)

        X[idxs] = X[idxs_rep[:no_elements[cl]]]

    return X


def get_dataset(name, path, tr_load_args = None, te_load_args = None):
    """
    Loads dataset

    Parameters
    ----------
    name: str
        Name of the dataset to be loaded. Supports MNIST and CIFAR10
    path: str
        Path to save the downloaded dataset
    tr_load_args: dict
        String dictionary for train distribution shift loading
    te_load_args: dict
        String dictionary for test distribution shift loading

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    if name == 'MNIST':
        return get_MNIST(path, tr_load_args, te_load_args)
    elif name == 'KMNIST':
        return get_KMNIST(path, tr_load_args, te_load_args)
    elif name == 'FASHION_MNIST':
        return get_FASHION_MNIST(path, tr_load_args, te_load_args)
    elif name == 'CIFAR10':
        return get_CIFAR10(path, tr_load_args, te_load_args)
    elif name == 'CIFAR100':
        return get_CIFAR100(path, tr_load_args, te_load_args)
    elif name == 'SVHN':
        return get_SVHN(path, tr_load_args, te_load_args)
    elif name == 'STL10':
        return get_STL10(path, tr_load_args, te_load_args)
    

def get_SVHN(path, tr_load_args = None, te_load_args = None):
    """
    Downloads SVHN dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 10
    
    # Download the SVHN dataset
    data_tr = datasets.SVHN(path + '/SVHN', split="train", download=True)
    data_te = datasets.SVHN(path + '/SVHN', split="test", download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data
    Y_tr = data_tr.labels
    X_te = data_te.data
    Y_te = data_te.labels
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_MNIST(path, tr_load_args = None, te_load_args = None):
    """
    Downloads MNIST dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 10
    
    # Download the MNIST dataset
    data_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    data_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data.numpy()
    Y_tr = data_tr.targets.numpy()
    X_te = data_te.data.numpy()
    Y_te = data_te.targets.numpy()
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_KMNIST(path, tr_load_args = None, te_load_args = None):
    """
    Downloads KMNIST dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 10
    
    # Download the KMNIST dataset
    data_tr = datasets.KMNIST(path + '/KMNIST', train=True, download=True)
    data_te = datasets.KMNIST(path + '/KMNIST', train=False, download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data.numpy()
    Y_tr = data_tr.targets.numpy()
    X_te = data_te.data.numpy()
    Y_te = data_te.targets.numpy()
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_FASHION_MNIST(path, tr_load_args = None, te_load_args = None):
    """
    Downloads FASHION_MNIST dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 10
    
    # Download the FASHION_MNIST dataset
    data_tr = datasets.FashionMNIST(path + '/FASHION_MNIST', train=True, download=True)
    data_te = datasets.FashionMNIST(path + '/FASHION_MNIST', train=False, download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data.numpy()
    Y_tr = data_tr.targets.numpy()
    X_te = data_te.data.numpy()
    Y_te = data_te.targets.numpy()
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path, tr_load_args = None, te_load_args = None):
    """
    Downloads CIFAR10 dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 10
    
    # Download the CIFAR10 dataset
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data
    Y_tr = data_tr.targets
    X_te = data_te.data
    Y_te = data_te.targets
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100(path, tr_load_args = None, te_load_args = None):
    """
    Downloads CIFAR100 dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 100
    
    # Download the CIFAR100 dataset
    data_tr = datasets.CIFAR100(path + '/CIFAR100', train=True, download=True)
    data_te = datasets.CIFAR100(path + '/CIFAR100', train=False, download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data
    Y_tr = data_tr.targets
    X_te = data_te.data
    Y_te = data_te.targets
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def get_STL10(path, tr_load_args = None, te_load_args = None):
    """
    Downloads STL10 dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Deterministic random seed to ensure data initialization is consistent
    np.random.seed(42)
    
    num_cls = 100
    
    # Download the STL10 dataset
    data_tr = datasets.STL10(path + '/STL10', split="train", download=True)
    data_te = datasets.STL10(path + '/STL10', split="test", download=True)
    
    # Obtain the raw data
    X_tr = data_tr.data
    Y_tr = data_tr.labels
    X_te = data_te.data
    Y_te = data_te.labels
    
    # Initialize tr_idx and te_idx, which contain the full list of indices. 
    # Used to select a subset from the the full dataset.
    tr_idx = [x for x in range(X_tr.shape[0])]
    te_idx = [x for x in range(X_te.shape[0])]
    
    # Prepare labels for subset selection
    Y_tr = np.array(Y_tr)
    Y_te = np.array(Y_te)
    
    # If the load arguments specify a class imbalance or a noise ratio, apply the distribution 
    # shift to the appropriate dataset. Note that only one of class imbalance or noise is applied.
    if tr_load_args is not None:
        if "class_imbalance_ratio" in tr_load_args:
            tr_idx = get_imbalanced_idx(Y_tr, num_cls, tr_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in tr_load_args:
            Y_tr = add_label_noise(Y_tr, num_cls, tr_load_args["noisy_labels_ratio"])

    if te_load_args is not None:
        if "class_imbalance_ratio" in te_load_args:
            te_idx = get_imbalanced_idx(Y_te, num_cls, te_load_args["class_imbalance_ratio"])
        elif "noisy_labels_ratio" in te_load_args:
            Y_te = add_label_noise(Y_te, num_cls, te_load_args["noisy_labels_ratio"])   
    
    # Select the subset specified by tr_idx and te_idx
    X_tr = X_tr[tr_idx]
    Y_tr = Y_tr[tr_idx]
    X_te = X_te[te_idx]
    Y_te = Y_te[te_idx]

    # Shuffle train and test datasets.
    train_permutation = np.random.choice(np.arange(len(Y_tr)), size=len(Y_tr), replace=False)
    test_permutation = np.random.choice(np.arange(len(Y_te)), size=len(Y_te), replace=False)
    X_tr = X_tr[train_permutation]
    Y_tr = Y_tr[train_permutation]
    X_te = X_te[test_permutation]
    Y_te = Y_te[test_permutation]

    # Convert labels to tensor
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)

    return X_tr, Y_tr, X_te, Y_te