import numpy as np
import torch
from torch.utils.data import Subset

def split_iid(dataset, num_clients):
    """
    Perform IID sharding of dataset across clients
    
    Args:
        dataset: Original dataset
        num_clients: Number of clients to split data
    
    Returns:
        List of client datasets with uniform distribution
    """
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    client_datasets = []
    
    # Shuffle indices to ensure random distribution
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    for i in range(num_clients):
        start = i * samples_per_client
        end = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
        client_indices = indices[start:end]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets

def split_non_iid(dataset, num_clients, classes_per_client):
    """
    Perform Non-IID sharding with controlled class distribution
    
    Args:
        dataset: Original dataset
        num_clients: Number of clients
        classes_per_client: Number of classes per client
    
    Returns:
        List of client datasets with non-uniform distribution
    """
    # Group samples by class
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Sort classes by number of samples
    sorted_classes = sorted(class_indices.keys(), 
                            key=lambda x: len(class_indices[x]), 
                            reverse=True)
    
    client_datasets = []
    
    for i in range(num_clients):
        client_indices = []
        
        # Select subset of classes for this client
        start_class = (i * classes_per_client) % len(sorted_classes)
        selected_classes = sorted_classes[start_class:start_class+classes_per_client]
        
        # Collect indices for selected classes
        for cls in selected_classes:
            # Distribute samples from each class
            class_sample_indices = class_indices[cls]
            split_size = len(class_sample_indices) // num_clients
            client_subset = class_sample_indices[
                i*split_size : (i+1)*split_size
            ]
            client_indices.extend(client_subset)
        
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets