import numpy as np
import torch.nn as nn
import torch

def One_Hot(class_id):
    class_ont_hot = np.zeros([20])
    class_ont_hot[class_id] = 1.0
    return class_ont_hot


mseloss = nn.MSELoss()
print(mseloss(torch.tensor(np.zeros([20])), torch.tensor(One_Hot(5))))





