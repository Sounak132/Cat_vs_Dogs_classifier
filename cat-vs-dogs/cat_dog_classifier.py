
# importing packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm

import skimage.io as io
from skimage.transform import resize
import random

import os



# Building a class for making data

class BuildDataset:
    def __init__(self, IM_SIZE, data_path, target_path):
        self.im_size = IM_SIZE
        self.data_path = data_path
        self.labels = {}
        self.data = []
        self.target_path = target_path
    def make_data(self):
        i = 0
        for Dir in os.listdir(self.data_path):
            if(Dir not in self.labels):
                self.labels[Dir] = i

            path = os.path.join(self.data_path, Dir)
            for file in tqdm(os.listdir(path)):
                cur_path = os.path.join(path, file)
                im = io.imread(cur_path, as_gray = True)
                im = resize(im, (self.im_size, self.im_size),anti_aliasing=True)
                dataPt = np.array([np.array(im), self.labels[Dir]])
                self.data.append(dataPt)
            i+=1
#         return (self.data, self.labels)
        np.save(self.target_path, self.data)



# Training_data

data_path = './input/dogs-vs-cats/train'
target_path = "./training_data.npy"


bs = BuildDataset(50, data_path, target_path)
bs.make_data()

#  testing_data
data_path = './input/dogs-vs-cats/test'
target_path = "./testing_data.npy"

bs = BuildDataset(50, data_path, target_path)
bs.make_data()



# helper functions for data processing and making tensors

def normalize(x):
    Max = torch.max(x)
    Min = torch.min(x)
    return x/(Max-Min)

def makeTensors(data):

    X = []
    y = []
    for i in range (len(data)):
        X.append(data[i][0])
        y.append(data[i][1])

    X = torch.unsqueeze(torch.tensor(X, requires_grad = False).float(), 1)
    y = torch.tensor(y, requires_grad = False)


    X = normalize(X)
    X.shape, y.shape
    return X, y

# loading training_data
training_data = np.load("./training_data.npy", allow_pickle = True)
np.random.shuffle(training_data)

X_train, y_train = makeTensors(training_data)




def accuracy(x, model, y,device):
    count =0
    pred = model(x.to(device))
    for i in range(len(y)):
        _, id = torch.max(pred[i], 0)
        if(id == y[i]):
            count+=1
    return count/len(y)




# Define the model

class Model(nn.Module):
    def __init__(self, x):
        super(Model, self).__init__()
        self.Shape = x.shape
        self.input_filters = self.Shape[1]
        self.flatten_input_shape = None

        self.cnv1 = nn.Conv2d(self.input_filters, 8, 2)
        self.cnv2 = nn.Conv2d(8, 32, 2)
        self.cnv3 = nn.Conv2d(32, 64, 3)
        self.cnv4 = nn.Conv2d(64, 128, 3)

        if(self.flatten_input_shape is None):
            self.forwardConv( torch.randn(1, self.input_filters, self.Shape[-2], self.Shape[-1]) )
        self.fc1 = nn.Linear(self.flatten_input_shape, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32,16)
        self.op = nn.Linear(16, 2)

    def forwardConv(self, x):
        x = F.max_pool2d(F.relu(self.cnv1(x)), 2)
        x = F.max_pool2d(F.relu(self.cnv2(x)), 2)
        x = F.max_pool2d(F.relu(self.cnv3(x)), 2)
        x = F.relu(self.cnv4(x))

        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

        if(self.flatten_input_shape is None):
            self.flatten_input_shape = x.shape[-1]

        return x

    def forward(self, x):
        x = self.forwardConv(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.op(x)

        return x

model = Model(X_train)
if(torch.cuda.is_available()):
    model.to('cuda')
summary(model, (1,50,50))



# traing function

def train(x, y, batch_size, epochs, model, optimizer, criterion, device):
    error_log = []
#     optimizer.to('cuda')
    for epoch in tqdm(range(epochs)):
        for i in range(0, x.shape[0], batch_size):
            X_batch = x[i:i+batch_size, :]
            y_batch = y[i:i+batch_size]

            if(device == 'cuda'):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            error_log.append(loss)
            loss.backward()

            optimizer.step()
#         print("{} : {}".format(epoch, loss))
    return error_log



# optimizer and loss function

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr= 1e-3)
criterion = nn.CrossEntropyLoss()


# training

error_log = train(X_train, y_train, 1000, 50, model, optimizer, criterion, "cuda")
print("accuracy :{}".format(accuracy(X_train, model, y_train, "cuda")))

# saving result

import matplotlib.pyplot as plt
plt.plot(error_log, 'bo')
plt.title("Result")
plt.xlabel("epochs")
plt.ylabel("error")
plt.savefig("result.jpg")


# saving model

torch.save(model, './model.pt')



# validation

test_data = np.load("testing_data.npy",allow_pickle = True)
X_test, y_test= makeTensors(test_data)

model = torch.load("./model.pt")
accuracy(X_test, model, y_test, "cuda")
