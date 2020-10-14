import unittest
import torch

from batchdist import __version__
from batchdist.src.batch_distance import BatchDistance

def test_version():
    assert __version__ == '0.1.0'

class SimpleExample():
    """
    Simple test example: computing sums along all non-batch dims 
    """
    def __init__(self):
        self.device = 'cpu'
        self.dtype = torch.float64
        # operation:
        self.op = self.operation

        # batched operation: 
        self.bop =  BatchDistance(
            self.op, 
            device=self.device, 
            dtype=self.dtype, 
            result_index=0
        )

    def operation(self, x,y):
        x_ = x.sum(axis=[1,2])
        y_ = y.sum(axis=[1,2])
        return x_ + y_

    def batched(self, x1,x2):
        return self.bop(x1, x2)

    def looped(self, x1, x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        D = torch.zeros(n1, n2, dtype=self.dtype)
        for i in range(n1):
            for j in range(n2):
                D[i,j] = self.op(x1[i:i+1], x2[j:j+1])[0]
        return D

class TestSimpleExample(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSimpleExample, self).__init__(*args, **kwargs)
        self.ex = SimpleExample()
        self._generate_data()
    
    def _generate_data(self):
        #generate data:
        self.x1 = torch.rand(10,4,3)
        self.x2 = torch.rand(10,4,3)
        self.x3 = torch.rand(8,4,3) # here different batch dim
        return self
 
    def _compare(self, x,y, pr=False):
        """
        function that checks that two inputs x,y are processed similarly. 
        """
        batched = self.ex.batched(x, y)
        looped = self.ex.looped(x, y)
        #print(f'batched value {batched}')
        #print(f'looped value {looped}')
        
        self.assertTrue(
                torch.equal(batched, looped)
        )

    def test_same(self, pr=True):
        self._compare(self.x1, self.x1)

    def test_different_same_size(self):
        self._compare(self.x1, self.x2)

    def test_different_varying_size(self):
        self._compare(self.x1, self.x3)

if __name__ == '__main__':
    unittest.main()
