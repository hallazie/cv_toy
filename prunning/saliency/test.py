# --*-- coding:utf-8 --*--

import numpy as np
import copy

def run():
    a = np.array([
    	[[4,4,4],[4,4,4],[4,4,4]],
    	[[2,2,2],[2,2,2],[2,2,2]],
    	[[1,1,1],[1,1,1],[1,1,1]],
 		[[5,5,5],[5,5,5],[5,5,5]],
    	[[3,3,3],[3,3,3],[3,3,3]],
    ])
    # print(a[[0,1,3],:,:])
    b = np.argsort(np.sum(a, axis=(1,2)))[::-1][:3]
    print(b)
    print(a[b,:,:])

class Model:
    def __init__(self):
        self.value = 10

    def dcopy(self, dec):
        t = copy.deepcopy(self)
        return t

if __name__ == '__main__':
    m = Model()
    print(m)
    p = m.dcopy(0)
    print(p)