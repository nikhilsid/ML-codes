#!/usr/bin/env python

import numpy as np

"""
Algorithm: (src: https://web.stanford.edu/class/cs168/l/l13.pdf)
Reservoir Sampling [Vitter ’85]
Given a number k, and a datastream x1, x2, . . . of length greater than k:
• Put the first k elements of the stream into a “reservoir” R = (x1, . . . , xk).
• For i ≥ k + 1
    – With probability k/i replace a random entry of R with x_i
• At the end of the stream, return the reservoir R.
"""

class Reservoir:
    def __init__(self, k, seed):
        self.k = k
        self.samples = []
        self.counter = 0
        np.random.seed(seed)

    def additem(self, newitem):
        self.counter += 1

        if len(self.samples) < self.k:
            self.samples.append(newitem)
            return

        if np.random.rand() <= self.k / self.counter:
            replacement_index = np.random.choice(self.k)
            self.samples[replacement_index] = newitem

    def getsamples(self):
        return self.samples

print('Simple test run with fixed seed')
r = Reservoir(5, 0)
for i in range(100):
    r.additem(i)
print(r.getsamples())

print('Repeated test run to see that it actually works')
from collections import defaultdict
d = defaultdict(int) # it means that default value is int(0)


for i in range(1_000_000): # repeating the exo a million times
    r = Reservoir(k=5, seed=i)  # sample of size=5
    for j in range(6): # stream of size=6
        r.additem(j)
    d[tuple(r.getsamples())] += 1

print({k:v/1_000_000 for k,v in d.items()})
# note that every tuple of samples has almost equal frequency (weighted frequency printed)



