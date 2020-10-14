import torch
import torch.nn as nn

class BatchDistance(nn.Module):
    """
    This module allows for the computation of a given 
    pair-wise operation (such as a distance) on the batch level.
    Notably, we assume that the operation already allows for a 
    batch dimension, however only computes the operation pair-wise.
    """
    def __init__(self, op, device='cpu', dtype=torch.float64, result_index=0):
        """
        - op: callable operation which takes two data batches x1,x2
            (both tensor of same dimensionality with the batch dimension
            in the first dimension), and returns a tensor of shape [batch_dim]
            with the pair-wise result of the operation.
            Example: for the operation f, op would return 
                [f(x1_1, x2_1), f(x1_2, x2_2), ..., f(x1_n, x2_n)] where n 
            refers to the batch dimension.
            In case, the op returns a tuple of results, result_index specifies
            which element refers to the result to use (by default 0, i.e. the first 
            element)
        """
        super(BatchDistance, self).__init__()
        self.op = op
        self.device = device
        self.dtype = dtype
        self.result_index = result_index

    def forward(self, x1, x2, **params):
        """
        computes batched distance operation for two batches of data x1 and x2
        (first dimension refering to the batch dimension) and returns the 
        matrix of distances.
        """
        # get batch dimension of both data batches
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        #if operation is computed on one dataset, we can skip redundant index pairs
        if x1.size() == x2.size() and torch.equal(x1, x2): 
            inds = torch.triu_indices(n1, n2)
            triu = True #use only upper triangular 
        else:
            inds = self._get_index_pairs(n1, n2) #get index pairs without looping
            triu = False
        # expand data such that pair-wise operation covers all required pairs of 
        # instances 
        x1_batch = x1[inds[0]]  
        x2_batch = x2[inds[1]]
        
        result = self.op(x1_batch, x2_batch)

        #check if op returns tuple of results and use the result_index'th element:
        if type(result) == tuple:
            result = result[self.result_index]

        #convert flat output to result matrix (e.g. a distance matrix)
        D = torch.zeros(n1, n2, dtype=self.dtype, device=self.device)
        D[inds[0], inds[1]] = result.to(dtype=self.dtype)
        if triu: #mirror upper triangular such that full distance matrix is recovered
            D = self._triu_to_full(D)
        return D 
    
    def _get_index_pairs(self, n1, n2):
        """
        return all pairs of indices of two 1d index tensors
        """ 
        x1 = torch.arange(n1)
        x2 = torch.arange(n2)

        x1_ = x1.repeat(x2.shape[0])
        x2_ = x2.repeat_interleave(x1.shape[0])
        return torch.stack([x1_, x2_])

    def _triu_to_full(self, D):
        """
        Convert triu (upper triangular) matrix to full, symmetric matrix.
        Assumes square input matrix D
        """
        diagonal = torch.eye(D.shape[0], 
                    dtype=torch.float64, 
                    device=self.device) 
        diagonal = diagonal * torch.diag(D) #eye matrix with diagonal entries of D 
        D = D + D.T - diagonal # add transpose minus diagonal values to convert 
        # upper triangular to full matrix 
        return D 
