#NNParams
#layerStructure defines the entire layout of the neural network
#for example, [3,5,1] is a neural network with 3 input nodes, 1 hidden layer with 5 nodes, and one output layer with one node
layerStructure = [4,12,12,1]
#right now, it seems that only sigmoid works as an activation function
#I haven't gotten around to fixing ReLU and tanh yet
actFunc = "sigmoid"

print("Defining Neural Network:")
print("Has layer structure: ", layerStructure)
print("Uses activation function: ", actFunc)
classBasedNeuralNetworkSig = cnn.NN([4,12,12,1],"sigmoid")

#define session storage file
#ISSUES:
#When Repl.it automatically saves all files, it only saves the file you are actually working on. If this script alters sessionFile.npy (which it will), those changes are not saved until you click on sessionFile.npy and repl.it thinks you are editing it
sessionFile = "sessionState.npy"

#loading data from previous session
#this is so you can have the same network you left off at. No traiing from scratch
if(os.path.isfile(sessionFile)):
  print("Loading all data previous session data from ", sessionFile)
  classBasedNeuralNetworkSig.importAllReleventData(np.load(sessionFile))

#Our X data. The idea of this network is that if any of the 4 bits is True, then the network returns True. (True being a number between 1 and 0 that rounds to 1)
print("Defining X,Y data:")
x = np.array([  [0,0,0,0],
                [0,0,0,1],
                [0,0,1,0],
                [0,0,1,1], 
                [0,1,0,0],
                [0,1,0,1],
                [0,1,1,0],
                [0,1,1,1],
                [1,0,0,0],
                [1,0,0,1],
                [1,0,1,0],
                [1,0,1,1], 
                [1,1,0,0],
                [1,1,0,1],
                [1,1,1,0],
                [1,1,1,1],
                ])
                  
y = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]).T

#train the network for nIter rounds
nIter = 100000
#print out progress after every x percent of iterations:
xPercent = 0.1
print("Training for N rounds: ", nIter)

doBackup = ""
if(nIter > 100000):
  doBackup = sessionFile

classBasedNeuralNetworkSig.trainXY(x,y,nIter,xPercent,doBackup)

print("Exporting all relevent session data:")
allReleventData = classBasedNeuralNetworkSig.exportAllReleventData()

print("Writing all relevent session data to ", sessionFile)
#write everything to a text file
np.save(sessionFile, allReleventData)

#this bit is just for comparisons sake. We compute all of X, and you can compare for yourself whether or not it is well enough trained
print()
print("Computing all X")
for a in x:
  tmp = classBasedNeuralNetworkSig.computeForX(a)[-1]
  print("X: ",a," -> ",tmp, "(rounded): ", np.round(tmp))
  
  
iterations = 100000
print("Running sigmoid")
sigmoidErrorValues = tnnl.baseline("sigmoid",iterations)
print("Running tanh")
tanhErrorValues = tnnl.baseline("tanh",iterations)
print("Running ReLU")
reluErrorValues = tnnl.baseline("relu",iterations)

sigAvgSlope = bio.raySlope(sigmoidErrorValues)
tanhAvgSlope = bio.raySlope(tanhErrorValues)
reluAvgSlope = bio.raySlope(reluErrorValues)

sigSemiDerivSlopes = bio.semiDerivSlope(sigmoidErrorValues)
tanhSemiDerivSlopes = bio.semiDerivSlope(tanhErrorValues)
reluSemiDerivSlopes = bio.semiDerivSlope(reluErrorValues)

print()
print("Sigmoid average rate of change in error (positive = increaseing error): ", sigAvgSlope)
print("Tanh average rate of change in error: (positive = increaseing error)", tanhAvgSlope)
print("ReLU average rate of change in error: (positive = increaseing error)", reluAvgSlope)
print()
'''
print("Sigmoid semi derivative slope values: ",sigSemiDerivSlopes)
print("Tanh semi derivative slope values: ",tanhSemiDerivSlopes)
print("Tanh semi derivative slope values: ",reluSemiDerivSlopes)
print()
print("Average of the semi derivative sigmoid slopes: ",bio.averageFltArray(sigSemiDerivSlopes))
print("Average of the semi derivative tanh slopes: ", bio.averageFltArray(tanhSemiDerivSlopes))
print("Average of the semi derivative tanh slopes: ", bio.averageFltArray(reluSemiDerivSlopes))
'''
#average the last p% values:
p = 0.0001
n = int(np.floor(iterations*p))

print("Average of the last",n,"error values for sigmoid: ", bio.averageFltArray(sigmoidErrorValues[-(n-1):]))
print("Average of the last",n,"error values for tanh: ", bio.averageFltArray(tanhErrorValues[-(n-1):]))
print("Average of the last",n,"error values for ReLU: ", bio.averageFltArray(reluErrorValues[-(n-1):]))
