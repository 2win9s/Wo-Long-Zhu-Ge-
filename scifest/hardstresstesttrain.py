import numpy as np
#set number of neurons here
neurons = np.zeros(shape = 10)

from datetime import datetime
import time

global NoOfRuns
NoOfRuns = 100

print (NoOfRuns,"cycles will be completed")

def time_convert(sec):
  mins = sec // 60
  minsb = round(mins)
  sec = sec % 60
  secb = round(sec,2)
  hours = mins // 60
  hoursb = round(hours)
  mins = mins % 60
  print("Completed in",hoursb,"hours",minsb, "mins and", secb, "seconds")

global CompRuns
CompRuns = 1

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
start_time = time.time()
print("Starting...", current_time)
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a  
  return reLUout  
input = np.zeros(shape = 1)
output = np.zeros(shape = 1)
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
      if memories[0][y - 1] !=  None:
          input[0] = input[0] + (fullnet[y] * memories[0][y - 1])
    fullnet[0] = reLU(input[0] + memoriesbias[0])
    for x in range(len(neurons)):
      y = x + 1
      for z in range(y+1,len(fullnet)):
        if memories[x][z - 1] != None:
          fullnet[y] = fullnet[y] + (fullnet[z] * memories[x][z - 1])
      for a in range(0,y):
        fullnet[y] = fullnet[y] + (fullnet[a] * memories[x][a])
      fullnet[y] = reLU(fullnet[y] + memoriesbias[y])
    for x in range(0,len(fullnet) - 1):
      if memories[-1][x] != None:
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
      if memories[a][sun] != None
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
def memoriesfall(l):
  global memoriesbias
  global placeholderz
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
        break
memoriesv1()
memoriesbiasv1()
startmemory()
sss = np.array([17 , 30 , 0 ,61, 68 , 4 , 8 ,19, 96, 96])
#sss = np.array([17 , 30 , 0 ,61, 68 , 4 , 8 ,19, 96, 96, 14,  7, 41, 61,  7]) 
#sss = np.array([17, 30 , 0 ,61 ,68,  4,  8, 19, 96, 96, 14,  7, 41, 61,  7, 68, 50, 29, 85, 30])
#sss = np.array([17 ,30 , 0, 61, 68 , 4 , 8 ,19 ,96 ,96, 14,  7 ,41, 61,  7, 68, 50, 29, 85, 30, 51, 56, 28, 41, 19]) 
#sss = np.array([17, 30,  0, 61 ,68 , 4 , 8, 19, 96, 96, 14,  7, 41, 61,  7, 68, 50, 29, 85, 30, 51, 56, 28, 41, 19, 47, 39, 79 ,22 , 2])
#sss = np.array([17 ,30 , 0 ,61 ,68  ,4  ,8 ,19 ,96 ,96 ,14  ,7 ,41 ,61 , 7 ,68 ,50 ,29 ,85 ,30 ,51 ,56 ,28, 41, 19,47 ,39 ,79 ,22 , 2, 65, 70 ,67 ,28 ,77])
#sss = np.arrray([17 ,30  ,0 ,61 ,68  ,4  ,8 ,19 ,96 ,96 ,14  ,7 ,41 ,61 , 7 ,68 ,50 ,29 ,85 ,30 ,51 ,56 ,28 ,41 ,19,47 ,39 ,79, 22  ,2, 65, 70, 67 ,28, 77, 87, 92 , 3, 23 ,45])
#sss = np..array([17 ,30  ,0 ,61 ,68  ,4  ,8 ,19 ,96 ,96, 14 , 7, 41, 61,  7 ,68, 50, 29 ,85 ,30 ,51 ,56 ,28 ,41 ,19,47 ,39, 79 ,22,  2 ,65 ,70 ,67 ,28 ,77 ,87 ,92 , 3 ,23, 45, 42, 74, 18 , 2 ,59])
for x in range(NoOfRuns):
  king = sss
  fear = np.random.randint(0,777)
  lear = np.append([fear],king)
  lear = np.append(lear,[fear])
  learnr = 0.0042579
  if x % 10 == 0 and x != 0:
    forget(0.0021)
    memorylink(4)
  for y in range(len(lear) - 1):
    input[0] = lear[y]
    target[0] = lear[y + 1]
    memoryactivationv1()
    hardcode()
    memorieslearn(learnr)
    memoriesfall(learnr)
    learnr = learnr * 0.69
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print ("Completed cycle", CompRuns , "of", NoOfRuns, "(",current_time,")")
  CompRuns += 1
  
  
with open("memories.p","wb") as ea:
    pickle.dump(memories,ea)
with open("memoriesbias.p","wb") as fa:
    pickle.dump(memoriesbias,fa)
  
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)
print ("Finished", NoOfRuns, "cycles at", current_time)
