import numpy as np
import sympy as sy
import math
import time

np.random.seed(123)

class NN():
  #non linear activation functions
  def nonlinSigmoid(self,x,deriv=False):
    if(deriv==True):
        #return (np.exp(x))/((np.exp(x)+1)**2)
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
  def nonlinTanh(self,x,deriv=False):
    if(deriv == True):
      return 1/(np.sech(x)**2)
    return np.tanh(x)
    
  def nonlinReLU(self,x,deriv=False):
    if(deriv):
      if(x>0):
        return 1
      else:
        return 0
    return np.maximum(0,x)
  
  def threshold(self,x,deriv=False):
    if(deriv):
      return ((x*nonlinReLU(self,x,deriv=True))-(nonlinReLU(self,x)*1))/(x**2)
    if(x==0):
      return 0
    else:
      return nonlinReLU(self,x)/x
  
  def softplus(self,x,deriv=False):
    if(deriv):
      return (np.exp(x))/(1 + np.exp(x))
    return np.log(1 + np.exp(x))
  
  #initilizer
  def __init__(self,inLayerStructure=[],actFunc="nonlinSigmoid"):
    self.layers = np.array([])
    self.synapses = np.array([])
    self.layerStructure = inLayerStructure
    self.formatLayerArray(self.layerStructure)
    self.randomlyAssignSynapses()
    if(actFunc == "nonlinSigmoid"):
      self.activationFunc = self.nonlinSigmoid
    if(actFunc == "nonlinTanh"):
      self.activationFunc = self.nonlinTanh
    if(actFunc == "nonlinReLU"):
      self.activationFunc = self.nonlinReLU
    if(actFunc == "threshold"):
      self.activationFunc = self.threshold
    if(actFunc == "softplus"):
      self.activationFunc = self.softplus
    #self.activationFunc = self.nonlinSigmoid
    #rounds spent training, in integers
    self.roundsTrained = 0
    #time spent training, in miliseconds
    self.milisTrained = 0
  
  #intilizer utility functions
  def formatLayerArray(self,layerStructure):
    tmp = []
    for a in range(len(layerStructure)):
      tmp.append([0]*layerStructure[a])
    self.layers = np.array(tmp)
  
  #we have to have some base network to improve upon, this creates it for us
  def randomlyAssignSynapses(self):
    #syn0 = 2*np.random.random((3,1)) - 1
    #3 = input nodes, 1 = output nodes
    tmp = []
    #tmp for 3,5,1
    #[[[[0.5 .1 .2 .3 .4][0.5 .1 .2 .3 .4][0.5 .1 .2 .3 .4]] [] []] []]
    for a in range(len(self.layers)-1):
      tmp.append(2*np.random.random((self.layerStructure[a],self.layerStructure[a+1])) - 1)
    self.synapses = np.array(tmp)
  
  #computational and training functions
  
  #compute NN for input X and return the n to last value (presuming you always want the last value, we use n = 0)
  def computeForX(self,x,rtnLast=True):
    #l1 = nonlin(np.dot(l0,syn0))
    #to compute a layer, take the nonlinear function (previous layer dot product connecting synapses)
    self.layers[0] = x
    for a in range(1,len(self.layers)):
      self.layers[a] = self.activationFunc(np.dot(self.layers[a-1],self.synapses[a-1]))
    if(rtnLast):
      return self.layers[-1]
    return self.layers
  
  def trainXY(self,x,y,iterations,defaultPercent=0.1,incrementalBackup="",doPrint=True):
    #define our various arrays
    #we have our layers, our synapses, our error values, and our delta values
    #layers and synapses are already defined
    
    #so technically lError[0] and lDelta[0] aren't needed. The inputs are always what the inputs should be, so these will never be used. They are included for the sake of simplicity, and having the same len() as layers[] and synapses[]
    if(len(incrementalBackup) > 0):
      doIncBackup = True
    else:
      doIncBackup = False
    startMilis = int(round(time.time() * 1000))
    for a in range(0,iterations):
      lError = []
      lDelta = []
      #feed forwards through all layers
      self.computeForX(x,rtnLast=False)
      
      #calculate and set the last layer's error
      lError.append(y - self.layers[-1])
      
      #l2_delta = l2_error*nonlin(l2,deriv=True)
      #calculate the last layer's delta 
      lDelta.append(lError[-1] * self.activationFunc(self.layers[-1]))
      
      #Fancy stuff starts now
      #l1_error = l2_delta.dot(syn1.T)
      #So basically we are going to try to calculate the previous layer's error from the current layer's delta and the connecting synapses
      #l1_delta = l1_error * nonlin(l1,deriv=True)
      
      for b in range(1,len(self.layers)):
        lError = [lDelta[-b].dot(self.synapses[-b].T)]+lError
        lDelta = [lError[-b-1] * self.activationFunc(self.layers[-b-1],deriv=True)]+lDelta
      
      #as far as i can tell, lError and lDelta are in good working order
      #print(lError)
      #print(lDelta)
      
      #so now we have to update all of our synapses and whatnot. iterably. FML.
      #syn1 += l1.T.dot(l2_delta)
      
      for c in range(0,len(self.synapses)):
        self.synapses[c] += self.layers[c].T.dot(lDelta[c+1])
      self.roundsTrained += 1
      #every 10%, print out our average error
      if(a%int((iterations*defaultPercent)) == 0):
        if(doPrint):
          print("***Rounds Trained: ",self.roundsTrained,". Average Error: ",np.mean(np.fabs(lError[-1])),"***")
        if(doIncBackup):
          np.save(incrementalBackup, self.exportAllReleventData())
        
        
    
    self.milisTrained += int(round(time.time() * 1000))-startMilis
    print("***Rounds Trained: ",self.roundsTrained,". Average Error: ",np.mean(np.fabs(lError[-1])),"***")
    print("***Total seconds spent training: ",self.milisTrained/1000," ***")
    print("***Average seconds spent per round calculated: ", np.array([float(self.milisTrained/self.roundsTrained/1000)]),"***")
    roundsTrained = 0

  #data storage functions
  #INCOMPLETE, partially functional
  def exportAllReleventData(self):
    return np.array([[self.layerStructure],[self.layers],[self.synapses],[self.activationFunc],[self.roundsTrained],[self.milisTrained]])
  
  def importAllReleventData(self,dataIn):
    self.layerStructure = dataIn[0][0]
    self.activationFunc = dataIn[3][0]
    self.roundsTrained = dataIn[4][0]
    self.synapses = dataIn[2][0]
    self.layers = dataIn[1][0]
    self.milisTrained = dataIn[5][0]
  
  def writeAllReleventData(self,path):
    np.save(path,self.exportAllReleventData)
    
  def readAllReleventData(self,path):
    self.importAllReleventData(np.load(path))
    
    
    
    
    