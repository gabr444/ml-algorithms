import matplotlib.pyplot as plt
import random

# Much like a feed forward layer, but without the activation function

class Algo():
    def __init__(self):
        # List with weights
        self.w = []
        # Bias value
        self.b = None
        
    # Linear regression
    def linear(self, x, typeRun="train"):
        if(len(self.w) == 0 and self.b is None):
            self.w = [random.uniform(-0.1, 0.1) for i in range(len(x))]
            self.b = random.uniform(-0.1, 0.1)
            
        output = []
        self.inputs = x
        self.out = 0
        for i in range(len(x)):
            self.out+=x[i]*self.w[i]
        self.out+=self.b
        
        return self.out

      # Mean squared error loss
    def mse(self, label):
        self.label = label
        # Calculate difference between label and output. Return squared difference
        return (self.label-self.out)**2

    def gradient(self, lr):
        # Derivative of mse function with respect to a weight = -2(x) where x is = label-predicted
        gradient = -2*(self.label-self.out)
        for i in range(len(self.inputs)):
            self.w[i]-=gradient*lr*self.inputs[i]
        self.b-=gradient*lr

algo = Algo()
# Input data
x = [[5, 6, 1, 2, 5], [5, 8, 1, 2, 6], [5, 10, 1, 2, 7]]
# Target data
y = [1, 2, 3]
# Learning rate (size of steps)
lr = 0.001
# Amount of times to train on entire dataset
epochs = 1000
print("training with", epochs, "epochs")  
for i in range(epochs):
    loss = 0
    for j in range(len(x)):
        out = algo.linear(x[j])
        loss+= algo.mse(y[j])
        algo.gradient(lr)

print("testing")
print("predicted, label")
for i in range(len(x)):
    print(algo.linear(x[i]), y[i])

