import numpy as np
#this python module is for the bare bones basic neural networks,no advanced stuff here
#notes for myself when you remove a neuron or a layer from hiddenlayers,input,or output update by running weightgen() biasgen() and/or targetgen()



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
    

#assigns a value y to element x of the input,note:lists start at 0 so first element is 0 2nd 1 one etc.
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
  k = None
  

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
            placeholder[x,y] = input[x] * weights[0][x,y]
    for x in range(len(placeholderz)):
        for y in range(len(placeholder)):
            placeholderz[x]  = placeholderz[x] + placeholder[y,x]
    hiddenlayers[0] = np.copy(placeholderz)
    for x in range(len(hiddenlayers)):
        y = x + 1
        if y < len(hiddenlayers):
            placeholder = np.copy(weights[y])
            placeholderz = np.zeros(shape = len(weights[y][0]))
            for a in range(len(hiddenlayers[x])):
                for b in range(len(hiddenlayers[y])):
                    placeholder[a,b] = hiddenlayers[x][a] * weights[y][a,b]
            for c in range(len(placeholderz)):
                for d in range(len(placeholder)):
                    placeholderz[c]  = placeholderz[c] + placeholder[d,c]
            hiddenlayers[y] = np.copy(placeholderz)
        else:
            placeholder = np.copy(weights[y])
            placeholderz = np.zeros(shape = len(output))
            for a in range(len(hiddenlayers[-1])):
                for b in range(len(hiddenlayers[y])):
                    placeholder[a,b] = hiddenlayers[-1][a] * weights[y][a,b]
            for c in range(len(placeholderz)):
                for d in range(len(placeholder)):
                    placeholderz[c]  = placeholderz[c] + placeholder[d,c]
            output = np.copy(placeholderz)
            y = None
    placeholder = None
    placeholderz = None
        
        




#get backpropagation going and we are done                                                   

for x in range(len(output)):
  global target
  target = []
  target.append(0)

def targetwrite(x,y):
  global target
  target[x]= y

def msecostgen():
  global output
  global target
  global cost
  global costly
  cost = 0
  costly = []
  for x in range(len(output)):
    costly.append((output[x] - target[x]) ** 2)
  for y in range(len(costly)):
    cost = cost + costly[x]
  k  = len(output)  
  cost  =  cost / k
  
  


def derivativereLU(x):                                         #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a


def backpropagation():
  global weights
  global output
  global hiddenlayers
  global bias
  global ouput
  global target
  global input
  global dereLU
  clear()
  global placeholder
  global placeholderz
  placeholder = []
  placeholderz  = []
  for x in range(len(bias)):
    placeholderz.append([])
    for y in range(len(bias[x])):
      placeholderz[x].append(0)
  for x in range(len(weights)):
    placeholder.append([])
    for y in range(len(weights[x])):
      placeholder[x].append([])
      for z in range(len(weights[x][y])):
        placeholder[x][y].append(0)
  
  for x in range(len(output)):
    crap = 0
    placeholderz[-1][x] = (output[x]*(1 - output[x])) * (2 * (output[x] - target[x]))* (1 / len(output)) + placeholderz[-1][x]
    
    #bit to be copied & pasted for each layer of bias starts here 
    
    for y in range(len (weights[-1][x])):
      crap = weights[-1][x][y] * (2 * (output[x] - target[x])) * (2 * (output[x] - target[x]))* (1 / len(output))
      for z in range(len(placeholderz[-2])):
        craap = 0
        derivativereLU(hiddenlayers[-1][z])
        placeholderz[-2][z] = dereLU * crap + placeholderz[-2][z]
    #bit to be copied & pasted for each layer of bias ends here
        
        for a in range(len(weights[-2][z])):
          craap = weights[-2][z][a] * dereLU * crap
          for b in range(len(placeholderz[-3])):
            craaap = 0
            derivativereLU(hiddenlayers[-2][b])
            placeholderz[-3][b] = dereLU * craap + placeholderz[-3][b]
                
            for c in range(len(weights[-3][b])):
                craaap = weights[-3][b][c] * dereLU * craap
                for d in range(len(placeholderz[0])):
                    craaaap = 0
                    derivativereLU(hiddenlayers[-3][d])
                    placeholderz[-4][d] = dereLU * craaap + placeholderz[-4][d]
  
  
  for x in range(len(hiddenlayers[-1])):
      for y in range(len(placeholder[-1][x])):
        shit = 0
        placeholder[-1][x][y] = hiddenlayers[-1][x] * (output[y]*(1 - output[y])) * (2 * (output[y] - target[y]))* (1 / len(output)) + placeholder[-1][x][y] 
        shit = weights[-1][x][y] * (output[y]*(1 - output[y])) * (2 * (output[x] - target[x]))* (1 / len(output))
        
        #bit to be copied & pasted for each layer of weights starts here
        
        for z in range(len(hiddenlayers[-2])):  
          for a in range(len(placeholder[-2][z])):
            shiit = 0
            derivativereLU(hiddenlayers[-1][a])
            placeholder[-2][z][a] = hiddenlayers[-2][z] * dereLU * shit + placeholder[-2][z][a]
            shiit = dereLU * weights[-2][z][a] * shit
            
            for b in range(len(hiddenlayers[-3])):
              for c in range(len(placeholder[-3][b])):
                shiiit = 0
                derivativereLU(hiddenlayers[-2][c])
                placeholder[-3][b][c] = hiddenlayers[-3][b] * dereLU * shiit + placeholder[-3][b][c]
                shiiit = dereLU * weights[-3][b][c]  * shiit
                
                for d in range(len(input)):
                    for e in range(len(placeholder[-4][d])):
                        shiiiit = 0
                        derivativereLU(hiddenlayers[-3][e])
                        placeholder[-4][d][e] = input[d] * dereLU * shiiit + placeholder[-4][d][e]
                        shiiiit = dereLU * weights[-4][d][e]  * shiiit
        
 #use as template copy and paste a feew times for multiple layers of weights and bises remember to switch around variables and schtuff like that
#"The best that most of us can hope to achieve in physics is simply to misunderstand at a deeper level." -- Wolfgang Pauli
#nothing  works !!!!!!!
#Sorry for being rubbish at coding
