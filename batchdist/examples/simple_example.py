import torch
from src.batch_distance import BatchDistance
from time import time

def dummy_distance(x,y):
    """
    This is a dummy distance d which allows for a batch dimension 
    (say with n instances in a batch), but does not return the full 
    n x n distance matrix but only a n-dimensional vector of the 
    pair-wise distances d(x_i,y_i) for all i in (1,...,n). 
    """
    x_ = x.sum(axis=[1,2])
    y_ = y.sum(axis=[1,2])
    return x_ + y_

def looped(f, x1, x2):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    D = torch.zeros(n1, n2)
    for i in range(n1):
        for j in range(n2):
            D[i,j] = f(x1[i:i+1], x2[j:j+1])
    return D


if __name__ == '__main__':
    #generate data (two batches of 10 samples of dimension [4,3])
    x1 = torch.rand(256,4,3)
    x2 = torch.rand(256,4,3)

    batched = BatchDistance(dummy_distance)
    
    start = time() 
    out1 = batched(x1, x2)
    print(f'batched took {time() - start} sec.')

    start = time() 
    # alternatively, using slow loops:
    out2 = looped(dummy_distance, x1, x2)
    print(f'looped took {time() - start} sec.')
    
