import torch
from torch import nn
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target, real_soft_label, real_noisy_labels


device = ('cuda' if torch.cuda.is_available() else 'cpu')

def noise(quantity, size):
    return Variable(torch.randn(quantity, size))

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, out_features, leakyRelu=0.2, hidden_layers=[128,256,512], in_features=64, escalonate = True):
        super(GeneratorNet, self).__init__()
        
        self.in_features = in_features
        self.layers = hidden_layers.copy()
        self.layers.insert(0, self.in_features)#权重初始化

        for count in range(0, len(self.layers)-1): # 3 hidden layers
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu)
                )
            )
        #add activation function architecture?n 输出层？
        if not escalonate:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features)
                )
            )
        else:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features),
                    escalonate #保证0-1范围内
                )
            )
    
    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x

    def create_data(self, quantity):
        points = noise(quantity, self.in_features)
        data = self.forward(points.cpu())
        return data

def train_generator(optimizer, discriminator, discriminator_aux, loss, fake_data, alpha, beta):

    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    loss_G_wgan = -torch.mean(prediction)
    prediction_aux = discriminator_aux(fake_data)
    loss_rawgan = loss(prediction_aux, real_noisy_labels(prediction_aux.size(0), 0.01))
    loss_G = alpha*loss_G_wgan + beta*loss_rawgan
    loss_G.backward()
    optimizer.step()
    return loss_G