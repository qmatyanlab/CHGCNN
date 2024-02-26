import numpy as np
import math


#define gaussian expansion for feature dimensionality expansion
class gaussian_expansion(object):
    def __init__(self, dmin, dmax, steps):
        assert dmin<dmax
        self.dmin = dmin
        self.dmax = dmax
        self.steps = steps-1
        
    def expand(self, distance, sig=None, tolerance = 0.01):
        drange = self.dmax-self.dmin
        step_size = drange/self.steps
        if sig == None:
            sig = step_size/2
        ds = [self.dmin + i*step_size for i in range(self.steps+1)]
        expansion = [math.exp(-(distance-center)**2/(2*sig**2))  for center in ds]
        expansion = [i if i > tolerance else 0 for i in expansion]
        return expansion