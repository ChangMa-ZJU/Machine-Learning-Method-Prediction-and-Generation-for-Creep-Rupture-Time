import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_treatment import DataSet, DataAtts
from discriminator import *
from generator import *
from utils import *

# multi_task regresssion
# Wasserstein GAN compare with vanilla GAN, 4 things have change：loss function without log; weight limit; last layer of discriminator without sigmoid；optimistic method；

# torch and pd mean function can't use
def keymean(z):
    sum = 0 
    av=0
    for i in range(len(z)):
        sum += z[i]
        av = sum /len(z)
    return av

# initial the weight of the network
def weights_init(m):
    classname = m.__class__.__name__                               
    if classname.find('Linear') != -1:                              
        nn.init.normal_(m.weight.data, 0.0, 0.02)                 
        nn.init.constant_(m.bias.data, 0)

# design the structure of the model(generator and discriminator and discriminator_aux)
class Architecture():
    def __init__(self, batch_size, learning_rate_d, learning_rate_d_aux, learning_rate_g, hidden_layers_d, hidden_layers_d_aux, hidden_layers_g, name):
        self.batch_size = batch_size
        self.learning_rate_d = learning_rate_d
        self.learning_rate_d_aux = learning_rate_d_aux
        self.learning_rate_g = learning_rate_g
        self.hidden_layers_d = hidden_layers_d
        self.hidden_layers_d_aux = hidden_layers_d_aux
        self.hidden_layers_g = hidden_layers_g
        self.name=name

#hyperparameter settings
num_epochs = [500] # 1000, 400, 300
batch_size = [50] # 100, 50, 5
number_of_experiments = 5

learning_rate_d = [0.00001] # i think learning rate should be tuned larger.
learning_rate_d_aux = [0.00001]
learning_rate_g = [0.00005]

# For different data types, you need to try a variety of different network structures.
hidden_layers_d = [[128, 64, 32]]  # [128, 64]descending order; or ascending order
hidden_layers_d_aux = [[256, 128, 64]]  #  [128, 64]In this process,the ability of discriminator is greater than generator
hidden_layers_g = [[64, 128, 256]]  # ascending size:[128, 256],  [256，512], [128，256，512]


# create the different architetures:generator and discriminator and discriminator_aux
architectures = []
count = 0
for lr_d in learning_rate_d:
    for lr_d_aux in learning_rate_d_aux:
        for lr_g in learning_rate_g:
            for b_size in batch_size:
                for hidden_d in hidden_layers_d:
                    for hidden_d_aux in hidden_layers_d_aux:
                        for hidden_g in hidden_layers_g:
                            for i in range(number_of_experiments):
                                name = str(count)
                                # name += "_" + str(i)
                                name += "_ba" + str(b_size)
                                name += "_ep" + str(num_epochs[0])
                                name += "_d" + str(len(hidden_d)) + ','.join(map(str, hidden_d))
                                name += "_da" + str(len(hidden_d_aux)) + ','.join(map(str, hidden_d_aux))
                                name += "_g" + ','.join(map(str, hidden_g))
                                name += "_lrd" + str(lr_d)
                                name += "_lrda" + str(lr_d_aux)
                                name += "_lrg" + str(lr_g)
                                                 
                                architectures.append(Architecture(
                                    batch_size=b_size,
                                    learning_rate_d=lr_d,
                                    learning_rate_d_aux=lr_d_aux,
                                    learning_rate_g=lr_g,
                                    hidden_layers_d=hidden_d,
                                    hidden_layers_d_aux=hidden_d_aux,
                                    hidden_layers_g=hidden_g,
                                    name=name
                                    )
                                )
                                count += 1

#training process
file_names = ["data/data_filter_nortar_trun.csv"] # "data/data_filter_nortar.csv", "data/data_nonfilter_nortar.csv"
esc = torch.nn.Sigmoid()  # The input of generator is normalized
loss = nn.BCELoss()  # loss function of two-class network
alpha, beta = 0.07, 0.3 # the loss of warsstein and rawGAN
for file_name, epochs in zip(file_names, num_epochs):
    dataAtts = DataAtts(file_name)
    database = DataSet(csv_file=file_name, root_dir=".", shuffle = False)
    for arc in architectures:

        generatorAtts = {
            'out_features': dataAtts.class_len,
            'leakyRelu': 0.2,
            'hidden_layers': arc.hidden_layers_g,
            'in_features': 100,
            'escalonate': esc #data is normalized
        }
        generator = GeneratorNet(**generatorAtts)
        generator.apply(weights_init)

        discriminatorAtts = {
            'in_features': dataAtts.class_len,
            'leakyRelu': 0.2,
            'dropout': 0.5,
            'hidden_layers': arc.hidden_layers_d  #[::-1] #descending order
        }
        discriminator = DiscriminatorNet(**discriminatorAtts)
        discriminator.apply(weights_init)

        discriminatorAtts_aux = {
            'in_features': dataAtts.class_len,
            'leakyRelu': 0.2,
            'dropout': 0.3,
            'hidden_layers': arc.hidden_layers_d_aux
        }
        discriminator_aux = DiscriminatorNet_aux(**discriminatorAtts_aux)
        discriminator_aux.apply(weights_init)

        # use GPU
        if torch.cuda.is_available():
            discriminator.cuda()
            discriminator_aux.cuda()
            generator.cuda()

        d_optimizer = optim.RMSprop(discriminator.parameters(), lr=arc.learning_rate_d)#it includes discriminator and discriminator_aux;
        da_optimizer = optim.Adam(discriminator_aux.parameters(), lr=arc.learning_rate_d_aux)
        g_optimizer = optim.Adam(generator.parameters(), lr=arc.learning_rate_g)
        #g_optimizer = optim.RMSprop(generator.parameters(), lr=arc.learning_rate_g)
        #g_optimizer = optim.SGD(generator.parameters(), lr=arc.learning_rate_g)
        #g_optimizer = optim.Adam(generator.parameters(), lr=arc.learning_rate_g)
        data_loader = torch.utils.data.DataLoader(database, batch_size=arc.batch_size, shuffle=False)
        num_batches = len(data_loader)
        print('The number of batches: {}'.format(num_batches))
        print('The structure of inputfile: {}'.format(dataAtts.fname))
        print('The structure of wholenetwork: {}'.format(arc.name))

        train_loss = []
        train_loss_g = []
        for epoch in range(epochs):
            if (epoch % 100 == 0):
                print("Epoch ", epoch)

            train_loss_batch = []
            train_loss_gbatch = []
            for n_batch, real_batch in enumerate(data_loader): 
                # 1.Train Discriminator 2.Train Generator.
                real_data = Variable(real_batch).float()
                if torch.cuda.is_available(): 
                    real_data = real_data.cuda()
                fake_data = generator(random_noise(real_data.size(0))).detach()
                dw_error = train_discriminator(d_optimizer, discriminator, real_data, fake_data)
                daux_error = train_discriminator_aux(da_optimizer, discriminator_aux, loss, real_data, fake_data)
                d_error = alpha*dw_error+beta*daux_error
                fake_data = generator(random_noise(real_batch.size(0)))
                g_error = train_generator(g_optimizer, discriminator, discriminator_aux, loss, fake_data, alpha, beta)
                train_loss_gbatch.append(g_error)
                train_loss_batch.append(d_error)

            train_loss.append(keymean(train_loss_batch))
            train_loss_g.append(keymean(train_loss_gbatch))

        savefig_path = './Figs/dual dis/' + '2/' + arc.name
        os.makedirs(savefig_path) # 创建多级文件夹

        x_train_loss = range(len(train_loss))
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.semilogy(x_train_loss, train_loss, linewidth=1.5)
        plt.title('The Training Loss Curve of D')
        plt.savefig(savefig_path + '/D_loss' + '.png', bbox_inches='tight')
        plt.clf()  # 画完前一张图后，将plt重置

        x_train_loss_g = range(len(train_loss_g))
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.semilogy(x_train_loss, train_loss_g, linewidth=1.5)
        plt.title('The Training Loss Curve of G')
        plt.savefig(savefig_path + '/G_loss' + '.png', bbox_inches='tight')
        plt.clf()  

        savemodel_path = './models/dual dis/' + '2/' + arc.name
        os.makedirs(savemodel_path)
        torch.save({
            'epoch': epoch,
            'model_attributes': generatorAtts,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            }, savemodel_path + '/gen.pt')

        torch.save({
            'epoch': epoch,
            'model_attributes': discriminatorAtts,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': d_optimizer.state_dict(),
        }, savemodel_path + '/dis.pt')

        torch.save({
            'epoch': epoch,
            'model_attributes': discriminatorAtts_aux,
            'model_state_dict': discriminator_aux.state_dict(),
            'optimizer_state_dict': da_optimizer.state_dict(),
        }, savemodel_path + '/dis_aux.pt')

