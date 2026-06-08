import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

#Create the model which will take in all weather metrics to establish training trends
#Output of the model will be linked to 5 categories
#No Snow, Low Snow (0.1-3), Medium Snow (3-8), High Snow (8-14), Extreme Snow (14+)
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features = 5):
        super().__init__() #Instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

#Manual seed for randomness and creating model instance
torch.manual_seed(41)
model = Model()

#Read data for snowfall and sort snow amount into defined categories