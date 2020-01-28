
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
        
def targetgen(x):
    global target
    target = np.zeros(shape = len(output))    
    
    
def targetwrite(x,y):
  global target
  target[x]= y
  

def msecostgen():   #not the most important formula
  global output
  global target
  global cost
  global costly
  cost = 0
  costly = np.zeros(shape = len(target))
  for x in range(len(target)):
    costly = ((output[x] - target[x]) ** 2)
  for y in range(len(costly)):
    cost = cost + costly[x]
  k  = len(output)  
  cost  =  cost / k
  
  
def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a
    

def backpropagationpt1():                                  #this is the important bit   
    global weights
    global output
    global hiddenlayers
    global bias
    global ouput
    global target
    global input
    global dereLU
    global placeholder
    global placeholderz
    placeholder = []
    placeholderz = []
    for x in range(len(weights)):
        placeholder.append(0)
        for y in range(len(weights[x])):
            placeholder[x] = np.zeros(shape = [len(weights[x]),len(weights[x][y])])
    for x in range(len(bias)):
        placeholderz.append(0)
        for y in range(len(bias[x])):
            placeholderz[x] = np.zeros(shape = bias[x].size)
    global crap
    crap = np.zeros(shape = len(weights))
    global stuff
    stuff = np.zeros(shape = len(bias))
    thing = len(weights)
    yoke = len(bias)
    for y in range(len(output)):
        placeholderz[-1][y] = (output[y] * (1 - output[y])) * (2 * (output[y] - target[y])) * (1 / len(output)) + placeholderz[-1][y]
        fish = yoke - 1
        stuff[fish] = (2 * (output[y] - target[y]))* (1 / len(output))* (output[y]*(1 - output[y]))
        backpropagationpt2(fish,y)
        for x in range(len(weights[-1])):
          placeholder[-1][x,y] = hiddenlayers[-1][x] * (output[y]*(1 - output[y])) * (2 * (output[y] - target[y]))* (1 / len(output)) + placeholder[-1][x,y] 
          notfish = thing - 1
          if weights[-1][x,y] != None:
            crap[notfish] = weights[-1][x,y] * (2 * (output[y] - target[y])) * (1 / len(output)) * (output[y] * (1 - output[y]))
            backpropagationpt3(notfish)

def backpropagationpt2(fish,y):
    global weights
    global output
    global hiddenlayers
    global bias
    global ouput
    global target
    global input
    global dereLU
    global stuff
    global placeholderz
    if fish == 1:
        for x in range(len(weights[fish])):
          if weights[fish][x,y] != None:
            stuff[fish] = stuff[fish] * weights[fish][x,y]
            derivativereLU(hiddenlayers[fish-1][x])
            placeholderz[fish-1][x] = dereLU * stuff[fish] + placeholderz[fish-1][x]
    else:
        for x in range(len(weights[fish])):
          if weights[fish][x,y] != None:
            stuff[fish] = stuff[fish] * weights[fish][x,y]
            derivativereLU(hiddenlayers[fish-1][x])
            placeholderz[fish-1][x] = dereLU * stuff[fish] + placeholderz[fish-1][x]
            stuff[fish-1] = stuff[fish] * dereLU
            k = fish - 1
            backpropagationpt2(k,x) 
                 
        
    
def backpropagationpt3(notfish):
    global weights
    global output
    global hiddenlayers
    global bias
    global ouput
    global target
    global input
    global dereLU
    global placeholder
    global crap
    if notfish == 1:
        for x in range(len(input)): 
            for y in range(len(hiddenlayers[notfish-1])):
              if input[x] != None:
                derivativereLU(hiddenlayers[notfish-1][y])
                placeholder[notfish-1][x,y] = input[x] * dereLU * crap[notfish] + placeholder[notfish-1][x,y]
    else:
        for x in range(len(hiddenlayers[notfish-2])): 
            for y in range(len(hiddenlayers[notfish-1])):
                derivativereLU(hiddenlayers[notfish-1][y])
                placeholder[notfish-1][x,y] = hiddenlayers[notfish-2][x] * dereLU * crap[notfish] + placeholder[notfish-1][x,y]
                if weights[notfish - 1][x,y] != None:
                  crap[notfish-1] = crap[notfish] * dereLU * weights[notfish-1][x,y]
                  trustmeitsnotfish = notfish - 1
                  backpropagationpt3(trustmeitsnotfish)

def backpropagationef():                                    #this one should be way more efficient in python as python isn't optimised for recursion but u need to cofigure the function in this module                                   
    print("i will eventually get around to writing backpropagationef, it is not necessary just more efficient as python is better at dealing with for loops (iterative) than repeatedly calling functions(recursive)")
    
def updateweights(l):
    global weights                                          #the arguement l is the learning rate, here we update the weights to minimize the cost
    global placeholder
    for x in range(len(weights)):
        for y in range(len(weights[x])):
            for z in range(weights[x][y].size):
              if weights[x][y,z] != None:
                weights[x][y,z] = weights[x][y,z] - (placeholder[x][y,z] * l)
    placeholder = None
                
def updatebias(l):
    global bias
    global placeholderz
    for x in range(len(bias)):
        for y in range(len(bias[x])):
            bias[x][y] = bias[x][y] - (placeholderz[x][y] * l)
    placeholderz = None
    
def diet(k):
   for x in range(len(weights)):
        for y in range(len(weights[x])):
            for z in range(weights[x][y].size):
              if weights[x][y,z] < k:
                weights[x][y,z] = None

# download pickles

import pickle
#input list of 18 subjects output 11 different points you cn get as a result pf your grade,[0,12,20,28,37,46,56,66,77,88,100], here we exclude the higher level + 25 points
nmofhl = [16,14]
inputgen(18)
hiddenlayersgen()
outputgen(11)
weightsgen()
biasgen()
eulerno(100000000000)
with open("traininput1.p","rb") as ab:
        traininput = pickle.load(ab)
with open("trainoutput1.p","rb") as bb:
        trainoutput = pickle.load(bb) 
with open("testinput1.p","rb") as cb:
        testinput = pickle.load(cb)
with open("testoutput1.p","rb") as db:
        testoutput = pickle.load(db)

sp = 0
heavyfat = 0
for fff in range(1120):
  heavyfat = heavyfat + 1
  if heavyfat % 250 == 0 and heavyfat!= 0:
    diet(0.05)
  if sp % 100 == 0:
    learnr = 0.020111812
  sp = sp + 1
  rrf = np.random.randint(0,len(trainoutput))
  input = traininput[rrf]
  targetgen(11)
  fireactivation()
  llm = trainoutput[rrf]
  if llm == 0:
    target[0] = 1
  if llm == 12:
    target[1] = 1
  if llm == 20:
    target[2] = 1
  if llm == 28:
    target[3] = 1
  if llm == 37:
    target[4] = 1
  if llm == 46:
    target[5] = 1
  if llm == 56:
    target[6] = 1
  if llm == 66:
    target[7] == 1
  if llm == 77:
    target[8] == 1
  if llm == 88:
    target[9] == 1
  if llm == 100:
    target[10] == 1
  backpropagationpt1()
  updateweights(learnr * 2)
  updatebias(learnr * 8)
  learnr = learnr * 0.9001
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
    elif dee < (testoutput[mmm] + 15) or dee > (testoutput[mmm] - 15):
        almost_right = almost_right + 1
print(correct)
print(almost_right)
with open("bias1.p","wb") as ea:
    pickle.dump(bias,ea)
with open("weights1.p","wb") as fa:
    pickle.dump(weights,fa)

                 
