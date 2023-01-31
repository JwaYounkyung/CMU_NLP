import numpy as np
import pandas as pd
import torch

a = np.array([[8,10]])

Ones_C   = np.ones((a.shape[1], 1), dtype="f")
softmax  = np.exp(a)/(np.exp(a)@Ones_C@np.transpose(Ones_C))
print(a)