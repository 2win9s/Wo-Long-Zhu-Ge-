# This is it, we are sticking with recursion multiprocessing will come later
import numpy as np
import pickle
import sys
import numba
import threading
import mpmath                    # mpf(float) this converts it to a custom presicison float i guess
miy =                            # adjust for performence targets basically precision (will only be use for backpropagation)
mp.dps = miy                     #set precision of mpf
kag = miy//2             
shiro = 1/(10 ** kag)            #will terminate branch of backpropagation if chain of prtial derivatives dips below this number(changes with prescision of mp.dps), reason being if this gets too small changes won't matter because of how small they are, the weights are only at most a longdouble
threading.stack_size(2 ** 27 - 1)#(around 17 mb, change if gpu has a lot of vram/need more heap or u need more stackspace, cpython stores only references in stack so 17mb should be enough)
sys.setrecursionlimit(7777777)   #change along with stack size and size of network& bptt/tpbtt depth
intern = np.zeros(shape =     )  #don't forget to initialise these
input = np.zeros(shape =     )   #rule of thumb have more inter neurons than input + output
output = np.zeros(shape =     )
synapselmt =                     #1.4 ** (the number of digits in total number of neurons - 1)
df =                             #sqrt of total number of neurons, if that is too much do cuberoot determines the curve for proportionalu
deviations=1.3                   # how many deviations to keep, remember empirical rule 68,95,99.7 
connectrate =                    # number should be between 0 and 0.1(keep it very small,fiddle around with the value to find a good oe), basically a multiplier on how many neurons to grow everytime connect runs, adjust to suit frequency of the connect functions 
weightmax =                      #this is the maximum value a weight should have (to prevent exploding weights)
                                 #remember to keep backup of fullnet for tbptt, also use random number of timesteps for each set of tbptt, not sure why but it seems to be a good practice
                                 #remember the format for targets is [[10],[100],[1000]...] and the format for target indice is [4,5,6...], len(target indice) = len(targets), indice tells us which part of the back up data we start from and apply our target
def proportionalu(x,lmt,c):      #(lmt sets the limit of what your function will converge to, c determines when the curve will get steep i.e. when stuff plateaus, not sure about how to exactly correlate the curve and c just use a reasonable dependant variable that is scaled to the number of neurons)
  out = (limit * x) / (x + c)
  return out
   
def setbase():
  global fullnet
  r = 1/ ((len(fullnet) - 1) * len(fullnet))
  out = np.e ** (np.log(r)/100)
  return out

def cregulator(x,base):          #the limit of setbase limits this output
  global fullnet
  global synapsec
  if x < synapsec:
    p = x / ((len(fullnet) - 1) * len(fullnet))
  else:
    p = (((x - synapsec)**2) + synapsec) /((len(fullnet) - 1) * len(fullnet))
  out = np.log(p) / np.log(base)
  return out


def reLU(x):
  if x > 0:
    reLUout = x
  else:
    reLUout = 0 
  return reLUout

def memories():
  global input
  global output
  global neurons
  global fullnet
  global memories
  damnitt = len(input) + len(intern) + len(output)
  fullnet = np.zeros(shape = damnitt)
  memories = []
  for x in range(len(fullnet)):
    memories.append(np.array([]))


def memoriesbias():
    global fullnet
    global memoriesbias
    memoriesbias = np.copy(fullnet)
    
def startmemory(ak,runs): not finished#ak starts the number of intital connections to and from input and output,set below 0.6, also take into account the , runs is the number of times connect() is run to randomly set up a few connections
  global memories
  global input
  global intern
  global output
  z = len(intern)
  y = len(input)
  xt = len(output)
  yz = y + z - 1
  yz1 = y + z
  y1 = y - 1
  percent = ak * z
  percent = percent // 1
  krr = percent
  if percent > y:
    percent = int(percent//y)
  else:
    percent = 1
  if krr > xt:
    krr = int(krr//xt)
  else:
    krr = 1
  neurin = np.array([])                      #indexes/indices? for next part of function
  for x in range(z):
    index = x + y
    neurin = np.append(neurin,[index], axis = None)
  neurin2 = np.copy(neurin)
  for inputnm in range(y):
    for x in range(len(percent)):
      resa = np.random.randint(0,len(neurin))
      if memories[neurin[resa]] == None:
        memories[neurin[resa]] = np.array([[inputnm,0]])
      else:
        memories[neurin[resa]] = np.append(memories[neurin[resa]],[[inputnm,0]], axis = 0)
      neurin = np.delete(neurin,resa)
  for x in range(xt):
    for y in range(krr):
      resa = np.random.randint(0,len(neurin2))
      if memories[x] == None:
        memories[x] = np.array([[neurin2[resa],0]])
      else:
        memories[x] = np.append(memories[x],[[neurin2[resa],0]], axis = 0)
      neurin2 = np.delete(neurin2,resa)
  for x in range(runs):
      connect()
  for x in range(len(memories)):
      part = ((2/len(memories[x])) ** 0.5)
      for y in range(len(memories[x])):
          xinit = np.random.randn()
          xinit = xinit * part
          memories[x][y,1] = xinit
     
def connect(): 
  global memories
  global fullnet
  global connectrate
  global cbase
  for x in range(fullnet):
      weightn = memories[x].size // 2
      if weightn < 1:
        weightn = 1
      mi = (len(fullnet) - weightn) * 0.01
      connectn   =   cregulator(weightn,cbase) * connectrate * mi
      if connectn >= 1:
          connectn = int(connectn)
      else:
        random = np.random.random_sample()
        if random <= connectn:
            connectn = 1
        else: 
            connectn = 0
      if connect != 0:
        list = np.array([])
        for y in range(x):
            list = np.append(list,[y])
        for z in range(x + 1 , len(fullnet)):
            list = np.append(list,[z])
        for t in range(connectn):
            neuron = np.random.randint(0,len(list))
            memories[x] = np.append(memories[x],[[list[neuron],0]],axis = 0)
            list = np.delete(list,neuron)
            xinit = np.random.randn()
            xinit = xinit * ((2/len(memories[x])) ** 0.5)
            memories[x][-1,1] = xinit
        
def disconnect():
  global fullnet
  global memories
  global deviations
  for x in range(len(fullnet)):
      population = memories[x].size // 2
      for y in range(population):
        if memories[x][y][1] < 0:
          mean += (memories[x][y][1] * -1)
        else:
          mean += memories[x][y][1]
      mean = mean / population
      for z in range(population):
          variance += ((memories[x][y][1] - mean) ** 2)
      variance = variance / population
      s_deviation = variance ** 0.5
      for z in range(population):
          if memories[x][z][1] < 0:
              absv = memories[x][z][1] * -1
          else:
              absv = memories[x][z][1]
          remains = mean - (s_deviation * deviations)
          if absv <= remains:
              memories[x] = np.delete(memories[x],z,0)

def memoryactivation():
    global fullnet
    global input
    global output
    global intern
    global memories
    global memoriesbias
    a = len(input)
    b = len(intern)
    c = len(output)
    e = a + b - 1
    f = a + b
    for x in range(fullnet):
      for y in range(memories[x]):
        fullnet[x] += fullnet[memories[x][y,0]] * memories[x][y,1]
      if x < a:
        fullnet[x] += input[x]
      fullnet[x] = reLU(fullnet[x] + bias[x])
      if fullnet[x] > e: 
        output[x - f] = fullnet[x]
    
  
#----------------------------------------------------------------------------------------------------------------

def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  if x > 0:
    dereLU = 1
  else:
    dereLU = 0
  return dereLU

def ba_zhen_tu(zhuge,targets,target_index):    #this is backpropagation
   global memories 
   global fullnet 
   global placeholder
   global placeholderz
   placeholder = []
   for x in range(len(memories)):
      placeholder.append(np.zeros(shape = len(memories[x])))
   placeholderz = np.zeros(shape = len(fullnet),dtype = np.longdouble) 
   for thing in range(len(target_index)): 
       sima = target_index[thing]
       target = targets[thing]
       hardcode(zhuge,target,sima) 

def hardcode(fullnet,target,sima): 
   global output 
   global placeholder 
   global placeholderz 
   global memories 
   global input
   global intern
   faker = sima - 1
   otuput = len(input) + len(intern)
   for ditto in range (len(output)): 
    same = otup + ditto
    rise = derivativereLU(fullnet[sima,same]) 
    finbar = (2 * (fullnet[sima,same] - target[ditto])) * rise 
    placeholderz[same] = mpf(finbar + placeholderz[same]) 
    for x in range(memories[same].size/2): 
        orn = memories[same][x,0]
        if same > orn:
                placeholder[same][x] =mpf(finbar * fullnet[sima,orn] + placeholder[same][x]) 
                rice = derivativereLU(fullnet[sima,orn]) 
                larry = finbar * rice
                placeholderz[orn] = mpf(larry + placeholderz[orn])
                larry = larry * memories[same][x,1]
                if rice != 0:
                    mario(orn,larry,fullnet,sima)
        else:
            placeholder[same][x] =mpf(finbar * fullnet[faker,orn] + placeholder[same][x]) 
            rice = derivativereLU(fullnet[faker,orn]) 
            larry = finbar * rice
            placeholderz[orn] = mpf(larry + placeholderz[orn])
            larry = larry * memories[same][x,1]
            if rice != 0:
                mario(orn,larry,fullnet,faker)
            
def mario(bbr,b,fin,al): 
   global memories 
   global placeholder       
   global placeholderz 
   global shiro
   for k in range(memories[bbr].size/2):
       taiga = memories[bbr][k][0]
       if taiga > bbr:
           if al != 0:
               ryuji = al - 1
               kill = 0
           else:
               kill = 0
       else:
           ryuji = al
           kill = 0
       if kill != 1:
           placeholder[bbr][taiga] +=mpf(b * fin[ryuji,taiga])
           harm = derivativereLU(fin[ryuji,taiga])
           taiping = harm * b
           placeholderz[ryuji] += mpf(taiping)
           taiping = peace * memories[bbr][taiga,1]
           if harm != 0:
               if taiping < 0:
                   r = taiping * -1
               else:
                   r = taiping
               if r > shiro:
                       r = None
                       mario(taiga,taiping,fin,ryuji)
           
            
            
def memorieslearn(l,ra):
    global memories
    global placeholder
    global memoriesbias
    global placeholderz
    global weightmax
    for x in range(len(memories)):
        for y in range(memories[x].size / 2):
            memories[x][y,1] += (placeholder[x][y] * l)
            if memories[x][y,1] > weightmax:
                memories[x][y,1] = weightmax
    for x in range(len(placeholder)):
        memoriesbias[x] += (placeholderz[x] * ra)
    placeholder = None
    placeholderz = None
          
    
memories()
startmemory()
memoriesbias()
synapsec = proportionalu((len(fullnet),synapselmt,df)
cbase = setbase()
startmemory(0.4,4)
