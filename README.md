# batchdist  

This is a small PyTorch-based package which allows for efficient batched operations, e.g. for computing distances without having to slowly loop over all instance pairs of a batch of data.

After having encountered mulitple instances of torch modules/methods promising to handling batches while only returning a vector of pairwise results (see example below) instead of the full matrix, this package serves as a tool to wrap such methods in order to return full matrices (e.g. distance matrices) using fast, batched operations (without loops). 

## Example  

First, let's define a custom distance function that only computes pair-wise distances for batches, so two batches of each 10 samples are 
converted to a distance vector of shape (10,).
```python  
>>> def dummy_distance(x,y):
        """
        This is a dummy distance d which allows for a batch dimension 
        (say with n instances in a batch), but does not return the full 
        n x n distance matrix but only a n-dimensional vector of the 
        pair-wise distances d(x_i,y_i) for all i in (1,...,n). 
        """
        x_ = x.sum(axis=[1,2])
        y_ = y.sum(axis=[1,2])
        return x_ + y_

# batchdist wraps a torch module around this callable to compute 
# the full n x n matrix with batched operations (no loops). 

>>> import batchdist as bd
>>> batched = bd.BatchDistance(dummy_distance)

# generate data (two batches of 256 samples of dimension [4,3])

>>> x1 = torch.rand(256,4,3)
>>> x2 = torch.rand(256,4,3)

>>> out1 = batched(x1, x2) # distance matrix of shape [256,256]
```
 
For more details, consult the included examples.

