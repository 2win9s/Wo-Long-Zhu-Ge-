import numpy as np
#THIS IS A TEST RUN FOR A WEIRD IDEA/ BACK UP PLAN
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a  
  return reLUout  
input = np.zeros(shape = 1)
input[0] = 2
output = np.zeros(shape = 1)
neurons = np.zeros(shape = 1)
def memoriesv1():
  global input
  global output
  global neurons
  global fullnet
  global memories
  fullnet = np.copy(input)
  fullnet = np.append(fullnet,neurons)
  fullnet = np.append(fullnet,output)
  memories = np.ones(shape = [len(fullnet),(len(fullnet ) - 1)])
memoriesv1()
def memoriesbiasv1():
    global fullnet
    global memoriesbias
    memoriesbias = np.copy(fullnet)
memoriesbiasv1()
def startmemory():
  global memories
  for x in range(len(memories)):
    for y in range(memories[x].size):
     rrr = np.random.randn()
     rrr = rrr * ((2/memories[x].size) ** 0.5)          
     memories[x,y] = rrr 
     
def memoryactivationv1():
    global fullnet
    global input
    global output
    global memories
    global neuronsbackup
    global memoriesbias
    global neurons
    global fullnetbackup
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
def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a
  return dereLU     
target = np.array([0])
    
def hardcode():
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
   placeholder = np.zeros(shape = [len(memories),(len(memories) - 1)])
   placeholderz = np.zeros(shape = len(fullnet))
   rise = derivativereLU(fullnet[-1])
   placeholderz[-1] = (2 * (fullnet[-1] - target[0])) * rise + placeholderz[-1]
   finbar = (2 * (fullnet[-1] - target[0])) * rise
   for x in range(memories[-1].size):
    larry = finbar * memories[-1][x]
    placeholder[-1][x] = (2 * (fullnet[-1] - target[0])) * rise * fullnet[x] + placeholder[-1][x]
    dice = (2 * (fullnet[-1] - target[0])) * rise * memories[-1][x]
    rice = derivativereLU(fullnet[x])
    placeholderz[x] = larry * rice + placeholderz[x]
    for y in range(memories[x].size):
      fishcat = larry * memories[x][y]
      placeholder[x][y] = dice * rice * fullnetbackup[y + 1] + placeholder[x][y]
      cise = dice * rice * memories[x][y]
      pice = derivativereLU(fullnetbackup[y + 1])
      placeholderz[y + 1]  = pice * fishcat + placeholderz[y + 1]
      for z in range(y):
        coyne = fishcat * memories[y][z]
        placeholder[y][z] = cise * pice * fullnetbackup[z] + placeholder[y][z]
        qice = cise * pice * memories[y][z]
        eice = derivativereLU(fullnetbackup[z])
        placeholderz[z] = eice * coyne + placeholderz[z]
        backpropagationpt1(z,qice,eice,coyne)
    for sun in range(x):
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
      wo_long_zhu_ge = d * memories[a][sun] 
      placeholder[a][sun] = b * c * fullnet[sun] + placeholder[a][sun]
      nice = b * c * memories[a][sun]
      sice = derivativereLU(fullnet[sun])
      placeholderz[sun] = wo_long_zhu_ge * sice + placeholderz[sun]
      backpropagationpt1(sun,nice,sice,wo_long_zhu_ge)
      backpropagationpt2(sun,nice,sice,wo_long_zhu_ge)
