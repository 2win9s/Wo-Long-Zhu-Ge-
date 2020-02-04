import numpy as np
# back propagation requires a copy of the previous neurons
#set number of neurons here
neurons = np.zeros(shape = 1)
input = np.zeros(shape = 1)
output = np.zeros(shape = 1)
target = np.copy(output) 

#-----------------------------------------------------------------------------------------------------------------------
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a  
  return reLUout  
  
def sigmoid(x):                                   #requires euler's number
  global sigout 
  eulern = 2.71828182845904523536028747135266249775
  var = 0.0072973525693                                     
  sigout = 1 / ( 1 + ( eulern ** ( var * x * (-1) ) ) )
  return sigout

#--------------------------------------------------------------------------------------------------------------------------------------------

def memories():
  global input
  global output
  global neurons
  global fullnet
  global memories
  fullnet = np.copy(input)
  fullnet = np.append(fullnet,neurons)
  fullnet = np.append(fullnet,output)
  memories = np.zeros(shape = [len(fullnet),(len(fullnet ) - 1)])

def memoriesbias():
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
     
def memoryactivation():
    global fullnet
    global input
    global output
    global memories
    global memoriesbias
    for x in range(len(fullnet)):
      for z in range(x + 1,len(fullnet)):
        if memories[x][z - 1] != None:
          fullnet[x] = fullnet[x] + (fullnet[z] * memories[x][z - 1])
      for a in range(0,x):
        fullnet[x] = fullnet[x] + (fullnet[a] * memories[x][a])
      if x < len(input):
        fullnet[x] = fullnet[x] + input[x]
      fullnet[x] = sigmoid(fullnet[x] + memoriesbias[x])
    for fish in range(len(output)):
      itsraw = (len(output) - fish) * -1
      output[itsraw] = fullnet[itsraw]
    
def netbackup(x,y):
  
#----------------------------------------------------------------------------------------------------------------

def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a
  return dereLU     
target = np.array([0])

def desig(x):
  global desig
  xx = x
  desig = sigmoid(xx) * ( 1 - sigmoid(xx))
  return desig

NOT FINISHED 

def RISE():
  global fullnet
  global input
  global output
  global memories
  global memoriesbias
  global fullnetbackup
  global placeholder
  global placeholderz
  placeholder = np.zeros(shape = [len(memories),(len(memories) - 1)])
  placeholderz = np.zeros(shape = len(fullnet))
  
  
  
  
  
def hardcode():
   global fullnet
   global input
   global output
   global memories
   global memoriesbias
   global fullnetbackup
   global placeholder
   global placeholderz
   global targetbackup
   global outputbackup
   placeholder = np.zeros(shape = [len(memories),(len(memories) - 1)])
   placeholderz = np.zeros(shape = len(fullnet))
   rise = desig(fullnet[-1])
   placeholderz[-1] = (2 * (fullnet[-1] - target[0])) * rise + placeholderz[-1]
   finbar = (2 * (fullnet[-1] - target[0])) * rise
   for x in range(memories[-1].size):
    if memories[-1][x] != None:
      larry = finbar * memories[-1][x]
      placeholder[-1][x] = (2 * (fullnet[-1] - target[0])) * rise * fullnet[x] + placeholder[-1][x]
      dice = (2 * (fullnet[-1] - target[0])) * rise * memories[-1][x]
      rice = derivativereLU(fullnet[x])
      placeholderz[x] = larry * rice + placeholderz[x]
      for y in range(memories[x].size):
        if memories[x][y] != None:
          fishcat = larry * memories[x][y]
          placeholder[x][y] = dice * rice * fullnetbackup[y + 1] + placeholder[x][y]
          cise = dice * rice * memories[x][y]
          pice = derivativereLU(fullnetbackup[y + 1])
          placeholderz[y + 1]  = pice * fishcat + placeholderz[y + 1]
          for z in range(y):
            if memories[y][z] != None:
              coyne = fishcat * memories[y][z]
              placeholder[y][z] = cise * pice * fullnetbackup[z] + placeholder[y][z]
              qice = cise * pice * memories[y][z]
              eice = derivativereLU(fullnetbackup[z])
              placeholderz[z] = eice * coyne + placeholderz[z]
              backpropagationpt1(z,qice,eice,coyne)
    for sun in range(x):
      if memories[x][sun] != None:
        firebar = finbar * memories[x][sun]
        placeholder[x][sun] = dice * rice * fullnet[sun] + placeholder[x][sun]
        nice = dice * rice * memories[x][sun]
        sice = derivativereLU(fullnet[sun])
        placeholderz[sun] = firebar * sice + placeholderz[sun]
        backpropagationpt1(sun,nice,sice,firebar)
        backpropagationpt2(sun,nice,sice,firebar)
      
def backpropagationpt1(a,b,c,d):
   global fullnet
   global input
   global output
   global memories
   global neuronsbackup
   global memoriesbias
   global neurons
   global fullnetbackup
   global placeholder  
   global placeholderz
   for k in range(a):
    if memories[a][k] != None:
      mr = d * memories[a][k]
      placeholder[a][k] = b * c * fullnetbackup[a] + placeholder[a][k]
      peace = b * c * memories[a][k]
      harm = derivativereLU(fullnetbackup[a])
      placeholderz[k] = mr * harm + placeholderz[k]
      backpropagationpt1(k,peace,harm,mr)

def backpropagationpt2(a,b,c,d):
   global fullnet
   global input
   global output
   global memories
   global neuronsbackup
   global memoriesbias
   global neurons
   global fullnetbackup
   global placeholder
   global placeholderz
   for sun in range(a):
      if memories[a][sun] != None:
        wo_long_zhu_ge = d * memories[a][sun] 
        placeholder[a][sun] = b * c * fullnet[sun] + placeholder[a][sun]
        nice = b * c * memories[a][sun]
        sice = derivativereLU(fullnet[sun])
        placeholderz[sun] = wo_long_zhu_ge * sice + placeholderz[sun]
        backpropagationpt1(sun,nice,sice,wo_long_zhu_ge)
        backpropagationpt2(sun,nice,sice,wo_long_zhu_ge)

def memorieslearn(l):
  global memories
  global placeholder
  for x in range(len(memories)):
    for y in range(memories[x].size):
      if memories[x,y] != None:
        memories[x,y] = memories[x,y] - (placeholder[x,y] * l)
    for x in range(len(memoriesbias)):
      memoriesbias[x] = memoriesbias[x] - (placeholderz[x] * l)

def forget(careful):
  global memories
  for x in range(len(memories)):
    for y in range(memories[x].size):
      if memories[x,y] != None:
        if memories[x,y] <= careful and memories[x,y] >= 0:
          memories[x,y] = None
        elif memories[x,y] >= (careful * -1) and memories[x,y] <= 0:
          memories[x,y] = None

def memorylink(please):
  global memories
  krusty = 0
  fish = 0
  while fish<= please:
    krusty = krusty + 1
    fire = np.random.randint(0,len(memories))
    crap = np.random.randint(0,memories[fire].size)
  if memories[fire][crap] == None:
      memories[fire][crap] = np.random.random_sample()
  elif krusty >= 31415926535900:
      pass #break i think this should be exit or pass. if not then just have nothing here. 
