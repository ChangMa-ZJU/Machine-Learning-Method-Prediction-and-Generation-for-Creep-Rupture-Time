import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import * 
from torch.autograd import grad as torch_grad

class DiscriminatorNet(torch.nn.Module):
    """
    torch.nn.Module：很重要的类，forward函数，.named_children函数 参数默认初始化吗？？
    A three hidden-layer discriminative neural network
    """
    def __init__(self, in_features, leakyRelu=0.2, dropout=0.3, hidden_layers=[1024, 512, 256]):
        super(DiscriminatorNet, self).__init__()
        
        out_features = 1
        self.layers = hidden_layers.copy()
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu),
                    nn.Dropout(dropout) #正则化
                )
            )
        
        self.add_module("out", 
            nn.Sequential(
                nn.Linear(self.layers[-1], out_features),
                #torch.nn.Sigmoid() #WGAN需要做出的改变不需要加sigmoid层
            )
        )

    def forward(self, x):
        for name, module in self.named_children(): #遍历所有直接子模块,不再递归下去,即子模块的子模块不会被遍历到
            x = module(x)
        return x


class DiscriminatorNet_aux(torch.nn.Module):
    """
    二分类网络binary-class network
    """
    def __init__(self, in_features, leakyRelu=0.2, dropout=0.3, hidden_layers=[1024, 512, 256]):
        super(DiscriminatorNet_aux, self).__init__()
        
        out_features = 1
        self.layers = hidden_layers.copy()
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu),
                    nn.Dropout(dropout) #正则化
                )
            )
        
        self.add_module("out", 
            nn.Sequential(
                nn.Linear(self.layers[-1], out_features),
                torch.nn.Sigmoid() #使用nn.BCELoss需要在该层前面加上Sigmoid函数，WGAN需要做出的改变
            )
        )

    def forward(self, x):
        for name, module in self.named_children(): #遍历所有直接子模块,不再递归下去,即子模块的子模块不会被遍历到
            x = module(x)
        return x


def calc_gradient_penalty(real_data, fake_data, netD):
    # make linear interpolation of real and fake data
    epsilon = torch.rand(real_data.size(0), 1)
    interpolated = epsilon * real_data.data + (1 - epsilon) * fake_data.data
    interpolated.requires_grad = True

    d_interpolated = netD(interpolated)

    gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(d_interpolated.size()),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    return ((gradients_norm - 1) ** 2).mean()


def train_discriminator(optimizer, discriminator, real_data, fake_data):  # discriminator,
    # Reset gradients
    optimizer.zero_grad()
    # Train on Real Data and Fake Data
    prediction_real = discriminator(real_data)
    prediction_fake = discriminator(fake_data)
    gradient_penalty = calc_gradient_penalty(real_data, fake_data, discriminator)
    # Calculate error and backpropagate
    loss_D = -torch.mean(prediction_real) + torch.mean(prediction_fake) + 12*gradient_penalty
    loss_D.backward()
    # Update weights with gradients
    optimizer.step()

    return loss_D

def train_discriminator_aux(optimizer, discriminator_aux, loss, real_data, fake_data):

    optimizer.zero_grad()
    prediction_real_aux = discriminator_aux(real_data)
    error_real_aux = loss(prediction_real_aux, real_noisy_labels(real_data.size(0), 0.01))
    prediction_fake_aux = discriminator_aux(fake_data)
    error_fake_aux = loss(prediction_fake_aux, fake_noisy_labels(real_data.size(0), 0.01))
    loss_D = error_real_aux + error_fake_aux
    loss_D.backward()
    optimizer.step()

    return loss_D