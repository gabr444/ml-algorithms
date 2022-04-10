import matplotlib.pyplot as plt
import random

class Linear():
    def __init__(self):
        # List with weights
        self.w = None
        # Bias value
        self.b = None
        self.linearType = None
        
    # Multiple Linear regression
    def multiLinear(self, x, typeRun="train"):
        self.linearType = "multiLinear"  
        if(self.w == None and self.b is None):
            self.w = [random.uniform(-0.1, 0.1) for i in range(len(x))]
            self.b = random.uniform(-0.1, 0.1)
        # Save inputs for gradient calculation
        self.inputs = x
        self.out = 0
        for i in range(len(x)):
            self.out+=x[i]*self.w[i]
        self.out+=self.b
        
        return self.out

    # Linear regression
    def linear(self, x):
        self.linearType = "linear"
        if(self.w == None and self.b is None):
            self.w = random.uniform(-0.1, 0.1)
            self.b = random.uniform(-0.1, 0.1)
        self.inputs = x
        self.out = 0
        self.out = (self.inputs*self.w)+self.b

        return self.out

    # Mean squared error loss
    def mse(self, label):
        self.label = label
        # Calculate difference between label and output and then square it. 
        return (self.label-self.out)**2

    def gradient(self, lr):
        # Derivative of mse function with respect to a weight = -2(x) where x is = label-predicted
        gradient = -2*(self.label-self.out)
        if(self.linearType == "multiLinear"):
            for i in range(len(self.inputs)):
                self.w[i]-=gradient*lr*self.inputs[i]
        
        elif(self.linearType == "linear"):
            self.w-=gradient*lr*self.inputs
        # Since the bias value is not dependent on input values we don't multiply it with the input
        self.b-=gradient*lr

    def getWeight(self):
        return self.w
    
    def getBias(self):
        return self.b

# Standard linear regression test (one parameter)
linear = Linear()

# Learning rate (size of steps)
lr = 0.01
# Amount of times to train on entire dataset
epochs = 20  

x = [1, 2, 3]
y = [3, 5, 7]

for i in range(epochs):
    loss=0
    for j in range(len(x)):
        out = linear.linear(x[j])
        loss+=linear.mse(y[j])
        linear.gradient(lr)
    print(loss)
    # Compare the labeled outputs to the current outputs on a graph (dots are labeled outputs, line is the current outputs).
    # You will notice how the line gets closer to the labeled outputs after each epoch.
    plt.plot(x, y, 'ro')
    plt.plot([x[0], x[len(x)-1]], [(x[0]*linear.getWeight())+linear.getBias(), (x[len(x)-1]*linear.getWeight())+linear.getBias()], 'r')
    plt.show()
    plt.close()

print("testing standard lr")
print("predicted, label")
for i in range(len(x)):
    print(linear.linear(x[i]), y[i])

# Input data
x = [[1, 1, 2], [1, 2, 3], [1, 3, 4]]
# Target data
y = [3, 5, 7]

# Multiple linear regression test
linear = Linear()

for i in range(epochs):
    loss = 0
    outputList = []
    for j in range(len(x)):
        out = linear.multiLinear(x[j])
        loss+= linear.mse(y[j])
        plt.plot()
        linear.gradient(lr)
    print(loss)

print("testing multiple lr")
print("predicted, label")
for i in range(len(x)):
    print(linear.multiLinear(x[i]), y[i])


