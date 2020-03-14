# This is it, we are sticking with recursion multiprocessing will come later
import numpy as np
import pickle
import sys
import threading
import mpmath            # mpf(float) this converts it
mp.dps = 100             # set precision of mpf
threading.stack_size(2 ** 27 - 1) #(around 17 mb, change if gpu has a lot of vram)
sys.setrecursionlimit(7777777)   #change along with stack size and stuff
intern = np.zeros(shape = None) #don't forget to initialise these
input = np.zeros(shape = None)   #rule of thumb have more inter neurons than input + output
output = np.zeros(shape = None)
synapselmt =                     #1.4 ** (the number of digits in total number of neurons - 1)
df =                             #sqrt of total number of neurons, if that is too much do cuberoot
#remember to keep backup of fullnet for tbptt
#remember the format for targets is [[10],[100],[1000]...] and the format for target indice is [4,5,6...], len(target indice) = len(targets), indice tells us which part of the back up data we start from and apply our target
def proportionalu(x,lmt,c):      #(lmt sets the limit of what your function will converge to, c determines when the curve will get steep)
  out = (limit * x) / (x + c)
  return out
   
def setbase(lmt):
  global fullnet
  r = 1/ ((len(fullnet) - 1) * len(fullnet))
  out = np.e ** (np.log(r)/lmt)
  return out

def proportionald(x,base):       #the limit of setbase limits this output
  global fullnet
  global synapsec
  if x < synapsec:
    p = x / ((len(fullnet) - 1) * len(fullnet))
  else:
    p = (((x - synapsec)**2) + synapsec) /((len(fullnet) - 1) * len(fullnet))
  out = np.log(p) / np.log(base)
  return out


#still need to create this and change the hell out of the code
def reLU(x):
  global reLUout
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
  global damnitt
  damnitt = len(input) + len(intern) + len(output)
  fullnet = np.zeros(shape = damnitt)
  memories = []
  for x in range(len(fullnet)):
    memories.append(np.array([]))


def memoriesbias():
    global fullnet
    global memoriesbias
    memoriesbias = np.copy(fullnet)
    
def startmemory(ak): not finished#ak starts the number of intital connections to and from input and output, set below 0.6 please, also take into account the number of input neurons and interneurons you have
  global memories
  global input
  global neurons
  global output
  z = len(neurons)
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
      resa = np.random.randint(0,len(neurin) - 1)
      if memories[neurin[resa]] == None:
        memories[neurin[resa]] = np.array([[inputnm,0]])
      else:
        memories[neurin[resa]] = np.append(memories[neurin[resa]],[[inputnm,0]], axis = 0)
      neurin = np.delete(neurin,resa)
  for x in range(xt):
    for y in range(krr):
      resa = np.random.randint(0,len(neurin2) - 1)
      if memories[x] == None:
        memories[x] = np.array([[neurin2[resa],0]])
      else:
        memories[x] = np.append(memories[x],[[neurin2[resa],0]], axis = 0)
      neurin2 = np.delete(neurin2,resa)
     

work on reconnect when you think about it
def connect(r): 

        
def cull(xs):


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
        fullnet[x] = fullnet[memories[x][y,0]] * memories[x][y,1] + fullnet[x]
      if x < a:
        fullnet[x] = fullnet[x] + input[x]
      fullnet[x] = reLU(fullnet[x] + bias[x])
      if fullnet[x] > e: 
        output[x - f] = fullnet[x]
    
  
#----------------------------------------------------------------------------------------------------------------

def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  global dereLU
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
   placeholder = np.zeros(shape = [len(memories),(len(memories) - 1)],dtype = np.longdouble) 
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
   faker = sima - 1
   for ditto in range (len(output)): 
    same = (ditto + 1 ) * -1 
    rise = derivativereLU(fullnet[sima,same]) 
    placeholderz[same] = (2 * (fullnet[sima,same] - target[same])) * rise + placeholderz[same] 
    finbar = (2 * (fullnet[sima,same] - target[same])) * rise 
    for x in range(memories[same].size): 
        if ((len(fullnet[sima]) - x )) > ditto + 1:
            if memories[same,x] != None: 
                larry = finbar * memories[same,x] 
                placeholder[same,x] = finbar * fullnet[sima,x] + placeholder[same,x] 
                dice = finbar * memories[same,x] 
                rice = derivativereLU(fullnet[sima,x]) 
                placeholderz[same] = larry * rice + placeholderz[same] 
                mario(x,dice,rice,larry,fullnet,sima) 
        else:
            if memories[same,x] != None: 
                next_three_subjects = x + 1
                larry = finbar * memories[same,x]
                placeholder[same,x] = finbar * fullnet[faker,x + 1] + placeholder[same,x] 
                dice = finbar * memories[same,x] 
                rice = derivativereLU(fullnet[faker,x + 1]) 
                placeholderz[same] = larry * rice + placeholderz[same] 
                mario(next_three_subjects,dice,rice,larry,fullnet,faker) 
def mario(bbr,b,c,d,fin,al): 
   global memories 
   global placeholder       
   global placeholderz 
   for k in range(len(memories[bbr])): 
        if memories[bbr,k] != None: 
           if k >= bbr:
                if al != 0:
                    ryuji = al - 1
                    taiga = k + 1
                    kill = 0
                else:
                    kill = 1
           elif bbr > k:
                ryuji = al
                taiga = k
                kill = 0
           if kill != 1:
             mr = d * memories[bbr,k] 
             placeholder[bbr,k] = b * c * fin[ryuji,taiga] + placeholder[bbr,k] 
             peace = b * c * memories[bbr,k] 
             harm = derivativereLU(fin[ryuji,taiga]) 
             placeholderz[bbr] = mr * harm + placeholderz[bbr] 
             mario(taiga,peace,harm,mr,fin,ryuji)
            
            
def memorieslearn(l,ra):
    global memories
    global placeholder
    global memoriesbias
    global placeholderz
    for x in range(len(memories)):
        for y in range(memories[x].size):
            if memories[x,y] != None:
                memories[x,y] = memories[x,y] - (placeholder[x,y] * l)
            if memories[x,y] > ra:
              memories[x,y] = ra
    for x in range(len(memoriesbias)):
        memoriesbias[x] = memoriesbias[x] - (placeholderz[x] * l)
    placeholder = None
    placeholderz = None
          
    
memories()
startmemory()
memoriesbias()
synapsec = proportionalu((len(neuron),synapselmt,df)
base = setbase(lmt)
