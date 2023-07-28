from torch.autograd.variable import Variable
import torch
from sklearn.preprocessing import MinMaxScaler
from numpy.random import choice

def random_noise(size):
    n = Variable(torch.randn(size, 100)) #100,64是generator的输入维数,
    #n = F.normalize(n, p=2, dim=1) #范数归一化
    scaler = MinMaxScaler()
    n = scaler.fit_transform(n)
    n = torch.from_numpy(n)
    n = n.to(torch.float32)
    if torch.cuda.is_available(): 
        return n.cuda() 
    return n

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1)) #返回全为1的tensor 真label取大，与tensorflow相反tf.keras.losses.BinaryCrossentropy
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1)) #return 0 tensor
    if torch.cuda.is_available(): return data.cuda()
    return data

def real_soft_label(size):
    '''real soft label'''
    data = Variable(torch.ones(size, 1)-0.2*torch.rand(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_soft_label(size):
    '''real fake label'''
    data = Variable(torch.zeros(size, 1)+0.2*torch.rand(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

# randomly flip some labels
def real_noisy_labels(data_size, p_flip):
	# determine the number of labels to flip
    data = Variable(torch.ones(data_size, 1)-0.2*torch.rand(data_size, 1))
    #data = Variable(torch.ones(data_size, 1))
    n_select = int(p_flip * data_size)
	# choose labels to flip
    flip_ix = choice([i for i in range(data_size)], size=n_select)
	# invert the labels in place
    data[flip_ix] = 1 - data[flip_ix]
    return data


def fake_noisy_labels(data_size, p_flip):
	# determine the number of labels to flip
    data = Variable(torch.zeros(data_size, 1)+0.2*torch.rand(data_size, 1))
    #data = Variable(torch.zeros(data_size, 1))
    n_select = int(p_flip * data_size)
	# choose labels to flip
    flip_ix = choice([i for i in range(data_size)], size=n_select)
	# invert the labels in place
    data[flip_ix] = 1 - data[flip_ix]
    return data