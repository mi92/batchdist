import torch
from src.batch_distance import BatchDistance
from example.external.sinkhorn_layer import SinkhornDistance

if __name__ == "__main__":
    device = 'cpu'
    dtype = torch.float64
    op = SinkhornDistance(
            eps=0.1, 
            max_iter=100, 
            reduction=None, 
            device=device
    )
    #generate data:
    x1 = torch.rand(10,4,3)
    x2 = torch.rand(10,4,3)
    
    #batched op:
    bop = BatchDistance(
            op, 
            device=device, 
            dtype=dtype, 
            result_index=0
    )
    r = bop(x1,x2)
    print(f'batched: {r}') 

