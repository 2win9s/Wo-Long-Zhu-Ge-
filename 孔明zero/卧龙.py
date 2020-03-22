import numpy as np
import sys
from numba import jit
from numba import prange
import dask
from datetime import datetime
import os
import time
import threading
from mpmath import mp 
import psutil
p = psutil.Process()
p.cpu_affinity([])
print ("cores",p.cpu_affinity())
print("There are maximum",(psutil.cpu_count(logical=True)),"threads concurrently running")
os.system("taskset -p 0xff %d" % os.getpid()) 
                                #mp.mpf(float) this converts it to a custom presicison float i guess
miy =  100                       # adjust for performence targets basically precision (will only be use for backpropagation)
mp.dps = miy                     #set precision of mpf
kag = miy//2             
shiro = 1/(10 ** kag)            #will terminate branch of backpropagation if chain of prtial derivatives dips below this number(changes with prescision of mp.dps), reason being if this gets too small changes won't matter because of how small they are, the weights are only at most a longdouble
threading.stack_size(2 ** 27 - 1)#(around 17 mb, change if gpu has a lot of vram/need more heap or u need more stackspace, cpython stores only references in stack so 17mb should be enough)
sys.setrecursionlimit(7777777)   #change along with stack size and size of network& bptt/tpbtt depth
intern = np.zeros(shape = 10)  #don't forget to initialise these
inputs = np.zeros(shape =  1  )  #rule of thumb have more inter neurons than input + output
output = np.zeros(shape =  1  )
synapselmt =5                  #1.4 ** (the number of digits in total number of neurons - 1)
df = 100                         #sqrt of total number of neurons, if that is too much do cuberoot determines the curve for proportionalu
deviations=1.3                   # how many deviations to keep, remember empirical rule 68,95,99.7 
connectrate =0.05                # number should be between 0 and 0.1(keep it very small,fiddle around with the value to find a good oe), basically a multiplier on how many neurons to grow everytime connect runs, adjust to suit frequency of the connect functions 
weightmax = 2                    #this is the maximum value a weight should have (to prevent exploding weights)
nlistp = len(intern) + len(inputs) + len(output)
nlist = np.zeros(shape = nlistp)

for x in range(nlistp):
    nlist[x] = x
    
def time_convert(sec):
  mins = sec // 60
  minsb = round(mins)
  sec = sec % 60
  secb = round(sec,2)
  hours = mins // 60
  hoursb = round(hours)
  mins = mins % 60
  print("Completed in",hoursb,"hours",minsb, "mins and", secb, "seconds")
  
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
start_time = time.time()
print("Starting...", current_time)
  
                               #remember to keep backup of fullnet for tbptt, also use random number of timesteps for each set of tbptt, not sure why but it seems to be a good practice
                 #remember the format for targets is [[10],[100],[1000]...] and the format for target indice is [4,5,6...], len(target indice) = len(targets), indice tells us which part of the back up data we start from and apply our target
def proportionalu(x,lmt,c):      #(lmt sets the limit of what your function will converge to, c determines when the curve will get steep i.e. when stuff plateaus, not sure about how to exactly correlate the curve and c just use a reasonable dependant variable that is scaled to the number of neurons)
  out = (lmt * x) / (x + c)
  return out
   

def setbase(x):
  r = 1/ ((x - 1) * x)
  out = np.e ** (np.log(r)/100)
  return out


def cregulator(x,base,fullnetsize,synapsec):          #the limit of setbase limits this output
  if x < synapsec:
    p = x / ((fullnetsize - 1) * fullnetsize)
  else:
    p = (((x - synapsec)**2) + synapsec) /((fullnetsize - 1) * fullnetsize)
  out = np.log(p) / np.log(base)
  return out


def reLU(x):
  if x > 0:
    reLUout = x
  else:
    reLUout = 0 
  return reLUout


def fullneting(inputs,intern,output):
  damnitt = len(inputs) + len(intern) + len(output)
  fullnet = np.zeros(shape = damnitt)
  return fullnet


def memorieslist(fullnet):
  memories= []
  for x in range(len(fullnet)):
    memories.append(None)
  return memories

def synapselist(fullnet):
    synapse = []
    for x in range(len(fullnet)):
        synapse.append(None)
    return synapse


def startmemory(runs,memories,connectrate,cbase,synapselmt,nlist):
    fno = len(memories)
    for x in range(len(memories)):
        if x == 0:
            z = len(memories) - 1
            memories[x] = np.array([z],dtype = int)
        else:
            r = x - 1
            memories[x] = np.array([r],dtype = int)
    for y in range(runs):
        memories = memoriesconnect(memories,connectrate,cbase,synapselmt,nlist,fno)
    return memories
 
      
def memoriesconnect(memories,connectrate,cbase,synapselmt,nlist,fno):
    for x in range(len(memories)):
        if type(memories[x]) == np.ndarray:
            number = memories[x].size
        else:
            number = 1
        mi = (fno - number) * 0.01
        connectn = cregulator(number,cbase,fno,synapselmt) * connectrate * mi
        if connectn > 1:
            connectn = int(connectn // 1)
        else:
            rn = np.random.random_sample()
            if rn < connectn:
                connectn = 1
            else:
                connectn = 0
        if connectn != 0:
            listi = np.copy(nlist)
            ppp = [x]
            if type(memories[x]) == np.ndarray:
                for y in range(len(memories[x])):
                    ppp.append(memories[x][y])
            listz = np.delete(listi,ppp)
            for r in range(connectn):
                if len(listz) != 0:
                    neuron = np.random.randint(0,len(listz))
                else:
                    break
                final_fish = np.array([listz[neuron]],dtype = int)
                if type(memories[x]) == np.ndarray:
                    memories[x] = np.append(memories[x],final_fish)
                else:
                    memories[x] = final_fish
    return memories
                    
def synapsegrow(synapse,memories):
    fish = len(memories)
    for x in range(fish):
        if type(synapse[x]) != np.ndarray:
            p = 0
        else:
            p = len(synapse[x])
        q = len(memories[x])
        pq = int(q - p)
        rtq = len(memories[x])
        for y in range(pq):
            init = np.random.randn()
            init *= ((2/rtq) ** 0.5)
            ainit = np.array([init],dtype=np.longdouble)
            if type(synapse[x]) != np.ndarray:
                synapse[x] = ainit
            else:
                synapse[x] = np.append(synapse[x],ainit)
    return synapse
                
            
def prune(deviations):
    global synapse
    global memories
    for x in range(len(synapse)):
        population = len(memories[x])
        mean = 0
        for y in range(population):
            if synapse[x][y] < 0:
                mean += (synapse[x][y] * -1)
            else:
                mean += synapse[x][y]
        mean = mean / population
        variance = 0
        for z in range(population):
            variance += ((synapse[x][z] - mean) ** 2)
        variance = variance / population
        eviation = variance ** 0.5
        devi = eviation * deviations
        remains = mean - devi
        rq = []
        for t in range(population):
            if synapse[x][t] < 0:
                absv = synapse[x][t] * -1
            else:
                absv = synapse[x][t]
            if absv <= remains:
                rq.append(t)
        synapse[x] = np.delete(synapse[x],rq)
        memories[x] = np.delete(memories[x],rq)

            




def memoriesbiasst(fullnet):
    memoriesbias = np.copy(fullnet)
    return memoriesbias
    


        


def memoryactivation(fullnet):
    global memories
    global inputs
    global synapse
    a = len(inputs)
    global memoriesbias
    for x in range(fullnet):
      for y in prange(memories[x]):
        fullnet[x] += fullnet[memories[x][y]] * synapse[x][y]
      if x < a:
        fullnet[x] += inputs[x]
      fullnet[x] = reLU(fullnet[x] + memoriesbias[x])
    return fullnet
      

def outputread(output):
    global fullnet
    for x in prange(len(output)):
        fish = (x + 1) * -1
        output[fish] = fullnet[fish]
    return output
        
  
#----------------------------------------------------------------------------------------------------------------

def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  if x > 0:
    dereLU = 1
  else:
    dereLU = 0
  return dereLU


def ba_zhen_tu(targets,target_index,output,fullnet,inputs,intern):    #this is backpropagation
   global placeholder
   placeholder = []
   for x in range(len(memories)):
      placeholder.append(np.zeros(shape = len(memories[x])))
   for x in range(len(placeholder)):
    for y in range(placeholder[x].size):
      placeholder[x,y] = mp.mpf(0.0)
   placeholder.append(np.zeros(shape = len(fullnet)))
   for x in range(len(placeholder[-1])):
      placeholder[-1][x] = mp.mpf(0.0)
   for thing in range(len(target_index)): 
       sima = target_index[thing]
       target = targets[thing]
       hardcode(target,sima,inputs,intern,output)



def hardcode(target,sima,inputs,intern,output): 
   global placeholder
   global zhuge
   global synapse
   global memories
   faker = sima - 1
   otup = len(inputs) + len(intern)
   for ditto in range (len(output)): 
    same = otup + ditto
    rise = derivativereLU(zhuge[sima,same]) 
    finbar = (2 * (zhuge[sima,same] - target[ditto])) * rise 
    placeholder[-1][same] = mp.mpf(finbar) + placeholder[-1][same]
    for x in range(memories[same].size): 
        orn = memories[same][x]
        if same > orn:
                placeholder[same][x]=(mp.mpf(finbar * zhuge[sima,orn])) + placeholder[same][x]
                rice = derivativereLU(zhuge[sima,orn]) 
                larry = finbar * rice
                placeholder[-1][orn]=placeholder[-1][orn]+(mp.mpf(larry))
                larry = larry * synapse[same][x]
                if rice != 0:
                    dask.delayed(mario)(orn,larry,sima)
                    placeholder = placeholder.compute()
        else:
            placeholder[same][x]=placeholder[same][x]+(mp.mpf(finbar * zhuge[faker,orn])) 
            rice = derivativereLU(zhuge[faker,orn]) 
            larry = finbar * rice
            placeholder[-1][orn]=placeholder[-1][orn]+(mp.mpf(larry))
            larry = larry * synapse[same][x]
            if rice != 0:
                dask.delayed(mario)(orn,larry,faker)
                placeholder = placeholder.compute()
   placeholder = placeholder.compute()
        





@dask.delayed
def mario(bbr,b,al): 
   global placeholder
   global zhuge
   global memories
   global synapse
   for k in range(memories[bbr].size):
           taiga = memories[bbr][k]
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
               placeholder[bbr][taiga]=placeholder[bbr][taiga]+(mp.mpf(b * zhuge[ryuji,taiga]))
               harm = derivativereLU(zhuge[ryuji,taiga])
               taiping = harm * b
               placeholder[-1][ryuji]+=(mp.mpf(taiping))
               taiping =taiping * synapse[bbr][taiga]
               if harm != 0:
                     mario(taiga,taiping,ryuji)

    

          


  

def memorieslearn(l,ra):
    global synapse
    global memories
    global placeholder
    global memoriesbias
    global weightmax
    for x in range(len(memories)):
        for y in range(memories[x].size / 2):
            synapse[x][y] -= (placeholder[x][y] * l)
            if synapse[x][y] > weightmax:
                syanpse[x][y] = weightmax
    for x in range(len(placeholder[-1])):
        memoriesbias[x] -= (placeholder[-1][x] * ra)
    placeholder = None 
fullnet = fullneting(inputs,intern,output)    
memories = memorieslist(fullnet)
synapse = synapselist(fullnet)
memoriesbias = memoriesbiasst(fullnet)
cbase = setbase(len(fullnet))
memories = startmemory(10,memories,connectrate,cbase,synapselmt,nlist)
