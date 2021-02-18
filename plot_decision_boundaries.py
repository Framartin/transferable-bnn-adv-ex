import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
sns.set_style("whitegrid")

from utils.data import CIFAR10
from numpy.random import random_sample


DIAMETER=10 # TODO

# import one example
data = CIFAR10(batch_size=32)
x = data.to_numpy(N=1)


# random direction
random_vec = random_sample(size=x.shape)
norm_rv = np.linalg.norm(random_vec, ord=2)
random_vec /= norm_rv  # random unit vect

# load x


