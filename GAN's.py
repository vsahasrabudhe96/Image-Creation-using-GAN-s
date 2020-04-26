from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import variable

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 0) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#Defining the generator
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100,out_channels = 512,kernel_size = 4,stride = 1,padding= 0,bias = False),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 512,out_channels = 256,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 256,out_channels = 128,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 128,out_channels = 64,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 64,out_channels = 3,kernel_size = 4,stride = 2,padding = 1,bias = False), #out_channels =3 because we want the Generator to generate images corresponding to the 3 channels i.e RGB
            nn.Tanh()
            )
    def forward(self, input):
        output = self.main(input)
        return output

#Creating the generator
netG = G() #since we only inherited the nn.Module and not input any parameters
netG.apply(weights_init)


#Defining the Discriminator
class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.LeakyReLU(negative_slope = 0.2,inplace = True),
            nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope = 0.2,inplace = True),
            nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope = 0.2,inplace = True),
            nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 4,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope = 0.2,inplace = True),
            nn.Conv2d(in_channels = 512,out_channels = 1,kernel_size = 4,stride = 1,padding = 0,bias = False),  #why 1?, bcoz the discriminator will be generating a single number as a single vector with dimension 1
            nn.Sigmoid()
            )
    def forward(self,input):  #will be the output of the generator
        output = self.main(input)
        return output.view(-1)   #flatten to change from 2d to 1d
    
#Creating the discriminator
netD = D() #since we only inherited the nn.Module and not input any parameters
netD.apply(weights_init)


#Training the network
criterion = nn.BCELoss()
optimD = optim.Adam(netD.parameters(),lr = 0.0002,betas = (0.5,0.999))
optimG = optim.Adam(netG.parameters(),lr = 0.0002,betas = (0.5,0.999))

for epoch in range(25):
    for i,data in enumerate(dataloader,start = 0):
        
        #1st step updating the weights of the neural network of the discriminator
        netD.zero_grad()
        
        #train the discriminator with real image
        real,_ = data #we dont want the labels here so thats why we are setting it to _
        input = variable(real)   #to convert the real image into a torch variable
        target = variable(torch.ones(input.size()[0]))  #we have to set targets to 1 as we are training the real image, so basically we have to create a torch array type of structure which will have 1's with the dimension equal to the minibatch size of real images
        output = netD(input) #forward pass through discriminator
        errD_real =criterion(output,target) #calc the real_errD loss
        
        #train the discriminator with fake image
        noise = variable(torch.randn(input.size()[0],100, 1, 1)) #1,1 stands for the dimension of each value
        fake = netG(noise) #forward pass through generator to generate the fake images
        target = variable(torch.zeros(input.size()[0])) #we have to set targets to 0 as we are training the fake image, so basically we have to create a torch array type of structure which will have 0's with the dimension equal to the minibatch size of real images
        output = netD(fake.detach())  #to remove the detach the gradients , so that no gradients are backpropagated through this variable
        errD_fake = criterion(output,target)
        
        #Backpropagation the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimD.step() #Applies the optimizer on the Neural network and updates the weights of the discriminator depending on how much it is responsible for total loss error
        
        
        #2nd step. Updating the weights of the neural network of the generator
        netG.zero_grad()
        target = variable(torch.ones(input.size()[0])) #here although we are generating the fake images, we will take the target as 1 because we want the discriminator to think that the fake images are real images
        output =netD(fake) #forward pass the fake images through the discriminator
        errG = criterion(output,target)
        errG.backward()
        optimG.step()
        print(errD.data)
        #3rd step:  Printing the losses and saving the real images and the generated image
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data, errG.data)) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        if i % 100 == 0: # Every 100 steps:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
            
        
