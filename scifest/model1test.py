import numpy as np
#reLU,efficient activation algorithm for hidden layers 
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a                                 #you can change this formula if you want something different where x is negative, you can study how different functions here affect things

def eulerno(x):                                   #note:run this function once to give a value to euler's number note:the larger x is the more accurate e will be
  global eulern                                   #euler's number or e
  eulern = (1+1/x)**x

def sigmoid(x):                                   #good output function for classsification problems with multiple outputs)
  global eulern                                   #requires euler's number
  global sigout 
  var = 0.1                                       
  sigout = 1 / ( 1 + ( eulern ** ( var * x * (-1) ) ) )

#neural network set up---------------------------------------------------------------------------------------
#creates a 1d numpy array of length x called input
def inputgen(x):
  global input
  input = np.zeros(shape = x)
    

#assigns a value y to element x of the input,note:lists start at 0 so first element is 0 2nd 1 one etc. basically useless
def inputwrite(x,y): 
  global input
  input[x] = y


#creates a list of 1d numpy arrays which acts as the hiddenlayers note:this function requires a list called nmofhl
def hiddenlayersgen():#if you want 3 hiddenlayers then nmofhl should contain three elements
  global nmofhl       #each element should be the number of neurons you want in that layer
  global hiddenlayers #note:hiddenlayers[x] gets the particular layer and hiddenlayers[x][y] gets the particular neuron
  hiddenlayers = []  
  for x in range(len(nmofhl)):
      hiddenlayers.append(np.zeros(shape = nmofhl[x]))
          
      
  
#creates a 1d numpy array of length x called output
def outputgen(x):
  global output
  output = np.zeros(shape = x)
  
#creates a list of 2d numpy arrays which act as weights in a neural network
def weightsgen():
  global weights
  global hiddenlayers
  global input
  global output                                    
  weights = []
  k = len(hiddenlayers)+1                          #there is one extra layer of weights because of output
  for x in range(k):                               
    weights.append(0)                              
  weights[0] = np.zeros(shape = [len(input),len(hiddenlayers[0])])  #the set of weights from input to first hidden layer               
  for x in range(len(hiddenlayers)):
     y = x + 1
     if y < len(hiddenlayers):                                      #adds weights to each row based on hidden layer 
        weights[y] = np.zeros(shape = [len(hiddenlayers[x]),len(hiddenlayers[y])])                     
     else:
        weights[-1] = np.zeros(shape = [len(hiddenlayers[-1]),len(output)]) #the set of weights from hiddenlayer to output
  for x in range(len(weights)):                      
    for y in range(len(weights[x])):
      for z in range(weights[x][y].size):
        rrr = np.random.randn()
        rrr = rrr * ((2/len(weights[x])) ** 0.5)          
        weights[x][y,z] = rrr
       #only choose rrr or sss not both to initiallize weights
       #sss = np.random.random_sample()
       #if sss < 0.5:
         #sss = sss - 1
       #weights[x][y,z] = sss'''

#creates a list of 1d numpy arrays which acts as the bias
def biasgen():
    global hiddenlayers
    global output
    global bias
    bias = hiddenlayers.copy()
    bias.append(np.zeros(shape = len(output)))



 #firing neurons/ forward pass--------------------------------------------------------------------------   
    
def fireactivation():
    global input
    global hiddenlayers
    global output
    global weights
    global bias
    placeholder = np.copy(weights[0])
    placeholderz = np.zeros(shape = len(weights[0][0]))
    for x in range(len(input)):
        for y in range(weights[0][x].size):
          if weights[0][x,y] != None:
            if input[x] != None:
              placeholder[x,y] = input[x] * weights[0][x,y]
    for x in range(len(placeholderz)):
        for y in range(len(placeholder)):
            placeholderz[x]  = placeholderz[x] + placeholder[y,x]
    for x in range(len(bias[0])):
        placeholderz[x]  = placeholderz[x] + bias[0][x]
        reLU(placeholderz[x])
        placeholderz[x] = reLUout
    hiddenlayers[0] = np.copy(placeholderz)
    for x in range(len(hiddenlayers)):
        y = x + 1
        if y < len(hiddenlayers):
            placeholder = np.copy(weights[y])
            placeholderz = np.zeros(shape = len(weights[y][0]))
            for a in range(len(hiddenlayers[x])):
                for b in range(len(hiddenlayers[y])):
                  if weights[y][a,b] != None:
                    placeholder[a,b] = hiddenlayers[x][a] * weights[y][a,b]
            for c in range(len(placeholderz)):
                for d in range(len(placeholder)):
                    placeholderz[c]  = placeholderz[c] + placeholder[d,c]
            for r in range(len(bias[y])):
                placeholderz[r]  = placeholderz[r] + bias[y][r]
                reLU(placeholderz[r])
                placeholderz[r] = reLUout
            hiddenlayers[y] = np.copy(placeholderz)
        else:
            placeholder = np.copy(weights[y])
            placeholderz = np.zeros(shape = len(output))
            for a in range(len(hiddenlayers[-1])):
                for b in range(len(output)):
                  if weights[y][a,b] != None:
                    placeholder[a,b] = hiddenlayers[-1][a] * weights[y][a,b]
            for c in range(len(placeholderz)):
                for d in range(len(placeholder)):
                    placeholderz[c]  = placeholderz[c] + placeholder[d,c]
            for s in range(len(bias[-1])):
                placeholderz[s]  = placeholderz[s] + bias[-1][s]
                sigmoid(placeholderz[s])
                placeholderz[s] = sigout
            output = np.copy(placeholderz)
import pickle
#input list of 18 subjects output 11 different points you cn get as a result pf your grade,[0,12,20,28,37,46,56,66,77,88,100], here we exclude the higher level + 25 points
nmofhl = [16,14]
inputgen(18)
outputgen(11)
hiddenlayersgen()
eulerno(100000000000)
with open("testinput1.p","rb") as cb:
        testinput = pickle.load(cb)
with open("testoutput1.p","rb") as db:
        testoutput = pickle.load(db)
with open("weights1.p","rb") as kke:
  weights = pickle.load(kke)
with open("bias1.p","rb") as rrp:
  bias = pickle.load(rrp)
correct = 0
almost_right = 0
for mmm in range(len(testinput)):
    input = testinput[mmm]
    fireactivation()
    fish = np.argmax(output)
    if fish == 0:
      dee = 0
    if fish == 1:
      dee = 12
    if fish == 2:
      dee = 20
    if fish == 3:
      dee = 28
    if fish == 4:
      dee = 37
    if fish == 5:
      dee = 46
    if fish == 6:
      dee = 56
    if fish == 7:
      dee = 66
    if fish == 8:
      dee = 77
    if fish == 9:
      dee = 88
    if fish == 10:
      dee = 100
    if dee == testoutput[mmm]:
       correct = correct + 1
    elif (dee <= (testoutput[mmm] + 15) and dee >= testoutput[mmm])or(dee >= (testoutput[mmm] - 15) and dee <= testoutput[mmm]):
        almost_right = almost_right + 1
print("correct")
print(correct)
print("almost right")
print(almost_right)
