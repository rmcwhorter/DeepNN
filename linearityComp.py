import numpy as np
import sympy as sy
import classNN as cnn

struct = [3,5,1]

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
Y = np.array([[0,1,1,0]]).T

sig = cnn.NN(inLayerStructure=struct)
linear = cnn.NN(inLayerStructure=struct,actFunc="lenActFunc")

sig.trainXY(X,Y,100)
print()
linear.trainXY(X,Y,100)