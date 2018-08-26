import numpy as np
import sympy as sy
import classNN as cnn
import os.path

#numpy printing options
print("Setting up Numpy printing options: ")
np.set_printoptions(formatter={'float': lambda x: format(x, '1.5E')})

structure = [2,1]

NANDX = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1]
])

NANDY = np.array([
  [1],
  [1],
  [1],
  [0]
])

#So we're trying to train a neural network that functions as a NAND gate
#Motivation is because Turing theorized that *any* computation could be preformed with the arbitrary application of NAND gates

nand = cnn.NN(inLayerStructure = structure,actFunc="threshold")
nand.trainXY(NANDX,NANDY,10000)