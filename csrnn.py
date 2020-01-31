import numpy as np
#THIS IS A TEST RUN FOR A WEIRD IDEA/ BACK UP PLAN
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a  
  return reLUout                                  #you can change this formula if you want something different where x is negative, you can study how different functions here affect things
#RUN THIS FUNCTION ONCE!!!
def eulerno(x):                                   #note:run this function once to give a value to euler's number note:the larger x is the more accurate e will be
  global eulern                                   #euler's number or e
  eulern = (1+1/x)**x

def sigmoid(x):                                   #good output function for classsification problems with multiple outputs)
  global eulern                                   #requires euler's number
  global sigout 
  var = 0.1                                       
  sigout = 1 / ( 1 + ( eulern ** ( var * x * (-1) ) ) )

#neural network set up---------------------------------------------------------------------------------------
#run these functions once and ONLY ONCE!!!!
#creates a 1d numpy array of length x called input
def inputgen(x):
  global input
  input = np.zeros(shape = x)
    

#assigns a value y to element x of the input,note:lists start at 0 so first element is 0 2nd 1 one etc.
def inputwrite(x,y): 
  global input
  input[x] = y


#creates a list of 1d numpy arrays which acts as the hiddenlayers note:this function requires a list called nmofhl
def neuronsgen(x):
  neurons = np.zeros(shape = x)
          
      
#creates a 1d numpy array of length x called output
def outputgen(x):
  global output
  output = np.zeros(shape = x)
  
  
def memoriesv1():
  global input
  global output
  global neurons
  global fullnet
  global memories
  fullnet = np.copy(input)
  fullnet = np.append(fullnet,neurons)
  fullnet = np.append(fullnet,output)
  memories = np.zeros(shape = [len(fullnet),(len(fullnet ) - 1)])
  

#creates a list of 1d numpy arrays which acts as the bias
def memoriesbiasv1():
    global fullnet
    global memoriesbias
    memoriesbias = np.copy(fullnet)

def startmemory():
  global memories
  for x in range(len(memories)):
    for y in range(memories[x].size):
     rrr = np.random.randn()
     rrr = rrr * ((2/memories[x].size) ** 0.5)          
     memories[x,y] = rrr 
     


 #firing neurons/ forward pass--------------------------------------------------------------------------   

count = 0


def input_record():
  global input
  global inputbackup
  global count
  if count!= 0 :
    inputbackup = np.copy(input)

def memoryactivationv1():
    global fullnet
    global input
    global output
    global memories
    global count
    global neuronsbackup
    global memoriesbias
    global neurons
    global fullnetbackup
    if count!= 0:
      fullnetbackup = np.copy(fullnet)
    for y in range(1,len(fullnet)):
          input[0] = input[0] + (fullnet[y] * memories[0][y - 1])
    fullnet[0] = reLU(input[0] + memoriesbias[0])
    for x in range(len(neurons)):
      y = x + 1
      for z in range(y+1,len(fullnet)):
        fullnet[y] = fullnet[y] + (fullnet[z] * memories[x][z - 1])
      for a in range(0,y):
        fullnet[y] = fullnet[y] + (fullnet[a] * memories[x][a])
      fullnet[y] = reLU(fullnet[y] + memoriesbias[y])
    for x in range(0,len(fullnet) - 1):
       fullnet[-1] = fullnet[-1] + (fullnet[x] * memories[-1][x])
    fullnet[-1] = reLU(fullnet[-1] + memoriesbias[-1])
    output[0] = fullnet[-1]
        
      
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
  return dereLU
    
#not finished yet
def backpropagationpt1():
  global fullnet
  global input
  global output
  global memories
  global count
  global neuronsbackup
  global memoriesbias
  global neurons
  global fullnetbackup
  global target
  placeholder = np.zeros(shape = [len(memories),memories[0].size])
  placeholderz = np.zeros(shape = [len(fullnet)])
  fire = np.zeros(shape = placeholderz.size - 1)
  fish = np.copy(fire)
  flame = np.copy(fish)
  for x in range(memories[-1].size):
    kek = (x+1) * -1
    ni = derivativereLU(fullnet[-1])
    placeholder[-1][kek] = (2 * (fullnet[-1] - target[0])) * ni * fullnet[kek - 1] + placeholder[-1][kek]
    fire[kek] = (2 * (fullnet[-1] - target[0])) * ni * memories * [-1][kek]
    for tr in range(0,(kek * -1)):
      fish = (tr + 1) * -1
      dev = derivativereLU(fullnet[kek - 1])
      placeholder[kek - 1][fish] = dev * fire[kek] * fullnetbackup[fish] + placeholder[kek - 1][fish]
      key = fire[kek] * dev * memories[kek - 1][fish]
      for dee in range((fish * - 1), len(fullnet)): 
        bob = (dee + 1) * -1
        def = derivaticereLU(fullnetbackup[fish])
        placeholder[fish][bob] = def * key * fullnetbackup[bob] + placeholder[fish][bob]
        nextkey = key * def * memories[fish][bob]
        backpropagationpt2(bob,nextkey)
    for re in range(((kek - 1) * -1),len(fullnnet):
        keam = (re + 1 ) * -1
        dud = derivativereLU(fullnet[kek - 1])
        placeholder[kek - 1][keam] = dud * fire[kek] * fullnet[keam] + placeholder[kek - 1][keam]
        lock = dud * fire[kek] * memories[kek - 1][keam]
        for dan in range(0,(keam * -1)):
          pap = (dan + 1) * -1
          flee = derivativereLU(fullnet[keam])
          placeholder[keam][dan] = flee * lock * fullnetbackup[pap] + placeholder[keam][dan]
          nextlock = flee * lock * memories[keam][dan]
          backpropagationpt2(dan,nextlock)
        backpropagationpt3(keam,lock)
  ri = derivativereLU(fullnet[-1])
  placeholderz[-1] = (2 * (fullnet[-1] - target[0])) * ri +  placeholderz[-1]
  con = (2 * (fullnet[-1] - target[0])) * ri
  for r in range(memories[-1].size):
    bleach = (x+1) * -1
    fish[bleach] = con * memories[-1][bleach]
    dereLU(fullnet[bleach 1]
    placeholderz[bleach] = 
    
def backpropagationpt2(bob,key):
  global fullnet
  global input
  global output
  global memories
  global count
  global neuronsbackup
  global memoriesbias
  global neurons
  global fullnetbackup
  global target
  for dee in range((bob * - 1), len(fullnet)):
    fll = (dee + 1) * -1
    def = derivaticereLU(fullnetbackup[bob])
    placeholder[bob][fll] = def * key * fullnetbackup[fll] + placeholder[bob][fll]
    newkey = key * def * memories[bob][fll]
    backpropagationpt2(fll,newkey)

def backpropagationpt3(p,l)
    for re in range(((p) * -1),len(fullnnet):
        keam = (re + 1 ) * -1
        dud = derivativereLU(fullnet[p])
        placeholder[p][keam] = dud * l * fullnet[keam] + placeholder[p][keam]
        lock = dud * l * memories[p][keam]
        for dan in range(0,(keam * -1)):
          pap = (dan + 1) * -1
          flee = derivativereLU(fullnet[keam])
          placeholder[keam][dan] = flee * lock * fullnetbackup[pap] + placeholder[keam][dan]
          nextlock = flee * lock * memories[keam][dan]
          backpropagationpt2(dan,nextlock)  
        backpropagationpt3(keam,lock)                            
                                      
                                      
                                      
                                      
                                      
