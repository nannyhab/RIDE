import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dynamic_sys_env import DynamicSystemEnv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

import os

class NeuralNetworkValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


env = DynamicSystemEnv()
ks = 27692.0
ks_normalized = (ks-env.ks_min)/(env.ks_max-env.ks_min)
cs = 1906.5
cs_normalized = (cs-env.cs_min)/(env.cs_max-env.cs_min)
s = env.reset(ks_normalized,cs_normalized)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

try:

    # Run on single GPU

    DEVICE_ID = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    os.system("clear")

except:

    # Run on CPU

    os.system("clear")

    print("No GPU found.")


df = pd.read_csv('Data_susp_CCD2.csv')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

lb_x = [-0.25,-2.5,-0.5,-2.5,env.ks_min,env.cs_min]; ub_x = [0.25,2.5,0.5,2.5,env.ks_max,env.cs_max]
model_initial = NeuralNetwork().to(device)
data = np.array(df)
input_data = data[:,0:6]
normalized_input_data = (input_data - np.min(input_data, axis=0, keepdims=True)) / (np.max(input_data, axis=0, keepdims=True) - np.min(input_data, axis=0, keepdims=True))
output_data = data[:,6]/1000

# Convert data to PyTorch tensors
inputs = torch.tensor(normalized_input_data, dtype=torch.float32).to(device)
targets = torch.tensor(output_data, dtype=torch.float32).reshape(-1, 1)
targets = targets.to(device)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(model_initial.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

num_epochs = 1000
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(input_data), batch_size):
        inputs_batch = inputs[i:i+batch_size]
        # Forward pass
        outputs = model_initial(inputs_batch)
        targets_batch = targets[i:i+batch_size]
        loss = criterion(outputs, targets_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Creating dataset
X1 = np.linspace(lb_x[0], ub_x[0], 2)
X2 = np.linspace(lb_x[1], ub_x[1], 2)
X3 = np.outer(np.linspace(lb_x[2], ub_x[2], 30), np.ones(30))
X4 = np.outer(np.linspace(lb_x[3], ub_x[3], 30), np.ones(30))
X4 = X4.T
ks_sample = np.linspace(lb_x[4], ub_x[4], 5)
cs_sample = np.linspace(lb_x[5], ub_x[5], 5)

Z = np.zeros((5,5,2,2,30,30))
for i1 in range(5):
    for i2 in range(5):
        for i in range(2):
            for j in range(2):
                for k in range(30):
                    for l in range(30):
                        inputs_tmp = np.array([X1[i],X2[j],X3[k,l],X4[k,l],ks_sample[i1],cs_sample[i2]])
                        inputs_tmp = (inputs_tmp - np.min(input_data, axis=0, keepdims=True)) / (np.max(input_data, axis=0, keepdims=True) - np.min(input_data, axis=0, keepdims=True))
                        normalized_inputs_tmp = torch.tensor(inputs_tmp.reshape(1,6), dtype=torch.float32).to(device)
                        Z_tmp = model_initial(normalized_inputs_tmp)*1000
                        Z[i1,i2,i,j,k,l] = Z_tmp.item()

# Creating figure
# fig = plt.figure(figsize =(14, 9))
# ax = plt.axes(projection ='3d')
 
# # Creating plot
# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_subplot(221, projection='3d')
# ax1.plot_surface(X3, X4, Z[1,1,0,0,:,:], alpha=0.75)
# ax1.title.set_text('x1='+'{0:.4f}'.format(X1[0])+', x2='+'{0:.4f}'.format(X2[0]))
# ax2 = fig.add_subplot(222, projection='3d')
# ax2.plot_surface(X3, X4, Z[1,1,1,0,:,:], alpha=0.75)
# ax2.title.set_text('x1='+'{0:.4f}'.format(X1[1])+', x2='+'{0:.4f}'.format(X2[0]))
# ax3 = fig.add_subplot(223, projection='3d')
# ax3.plot_surface(X3, X4, Z[1,1,0,1,:,:], alpha=0.75)
# ax3.title.set_text('x1='+'{0:.4f}'.format(X1[0])+', x2='+'{0:.4f}'.format(X2[1]))
# ax4 = fig.add_subplot(224, projection='3d')
# ax4.plot_surface(X3, X4, Z[1,1,1,1,:,:], alpha=0.75)
# ax4.title.set_text('x1='+'{0:.4f}'.format(X1[1])+', x2='+'{0:.4f}'.format(X2[1]))
# plt.suptitle('Initial policy for ks = '+str(ks_sample[1])+', cs = '+str(cs_sample[1]))
# # Show plot
# plt.show()

# # Creating plot
# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_subplot(221, projection='3d')
# ax1.plot_surface(X3, X4, Z[3,3,0,0,:,:], alpha=0.75)
# ax1.title.set_text('x1='+'{0:.4f}'.format(X1[0])+', x2='+'{0:.4f}'.format(X2[0]))
# ax2 = fig.add_subplot(222, projection='3d')
# ax2.plot_surface(X3, X4, Z[3,3,1,0,:,:], alpha=0.75)
# ax2.title.set_text('x1='+'{0:.4f}'.format(X1[1])+', x2='+'{0:.4f}'.format(X2[0]))
# ax3 = fig.add_subplot(223, projection='3d')
# ax3.plot_surface(X3, X4, Z[3,3,0,1,:,:], alpha=0.75)
# ax3.title.set_text('x1='+'{0:.4f}'.format(X1[0])+', x2='+'{0:.4f}'.format(X2[1]))
# ax4 = fig.add_subplot(224, projection='3d')
# ax4.plot_surface(X3, X4, Z[3,3,1,1,:,:], alpha=0.75)
# ax4.title.set_text('x1='+'{0:.4f}'.format(X1[1])+', x2='+'{0:.4f}'.format(X2[1]))
# plt.suptitle('Initial policy for ks = '+str(ks_sample[3])+', cs = '+str(cs_sample[3]))
# # Show plot
# plt.show()
ks = 20000.0
ks_normalized = (ks-env.ks_min)/(env.ks_max-env.ks_min)
cs = 3000.0
cs_normalized = (cs-env.cs_min)/(env.cs_max-env.cs_min)

s = env.reset(ks_normalized,cs_normalized)
print("Initial state:[{:.4f},{:.4f},{:.4f},{:.4f}]".format(s[0],s[1],s[2],s[3]))
score = 0
x_sys = np.zeros((4,101))
x_sys[:,0] = s
a_sys = np.zeros((100,))

for i in range(100):
    # mu = model(torch.from_numpy(s).float())
    inputs_tmp = np.array([s[0],s[1],s[2],s[3],ks,cs])
    normalized_inputs_tmp = (inputs_tmp - np.min(input_data, axis=0, keepdims=True)) / (np.max(input_data, axis=0, keepdims=True) - np.min(input_data, axis=0, keepdims=True))
    normalized_inputs_tmp = torch.tensor(normalized_inputs_tmp.reshape(1,6), dtype=torch.float32).to(device)
    mu = model_initial(normalized_inputs_tmp)*1000
    dist = Normal(mu, torch.tensor(10).to(device)) # std = 10
    a = dist.sample()
    a = min(max(a.item(), env.action_space.low[0]), env.action_space.high[0])
    # a = min(max(a.item(), -500), 500)
    a_sys[i] = a
    s_prime, r, done, _ = env.step(a,ks_normalized,cs_normalized)
    x_sys[:,i+1] = s_prime
    score += r
    print("Current state:[{:.4f},{:.4f},{:.4f},{:.4f}], Action: {:.4f}, Reward: {:.4f}, Next state:[{:.4f},{:.4f},{:.4f},{:.4f}]".format(s[0],s[1],s[2],s[3],a,r,s_prime[0],s_prime[1],s_prime[2],s_prime[3]))
    s = s_prime
print("Final score:{:.4f}".format(score))

