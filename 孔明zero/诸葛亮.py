import numpy as np
# back propagation requires a copy of the previous neurons
#set number of neurons here
import sys
import threading
import psutil
p = psutil.Process()
p.cpu_affinity([])
threading.stack_size(2 ** 27 - 1)   #(around 17 mb,shoud be enough
sys.setrecursionlimit(7777777)      #change along with stack size and size of neuralnet & bptt/tpbtt depth
#-----------------------------------------------------------------------------------------------------------------------
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.001                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
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
  global damnitt
  damnitt = len(input) + len(neurons) + len(output)
  fullnet = np.zeros(shape = damnitt)
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
        if memories[x,z - 1] != None:
          fullnet[x] = fullnet[x] + (fullnet[z] * memories[x,z - 1])
      for a in range(0,x):
        if memories[x,a] != None:
          fullnet[x] = fullnet[x] + (fullnet[a] * memories[x,a])
      if x < len(input):
        fullnet[x] = fullnet[x] + input[x]
      fullnet[x] = reLU(fullnet[x] + memoriesbias[x])
    for fish in range(len(output)):
      itsraw = (len(output) - fish) * -1
      output[itsraw] = fullnet[itsraw]
    
  
#----------------------------------------------------------------------------------------------------------------

def derivativereLU(x):                                     #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a
  return dereLU     


def desig(x):
  global desig
  xx = x
  desig = sigmoid(xx) * ( 1 - sigmoid(xx))
  return desig

   
  
  
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
       hardcode(target,sima) 

def hardcode(target,sima): 
   global placeholder
   global zhuge
   global memories
   global input
   global neurons
   global placeholderz
   global output
   t = len(input) + len(neurons)
   for ditto in range (len(output)): 
    same = t + ditto
    rise = derivativereLU(zhuge[sima,same])
    finbar = 2 * (zhuge[sima,same] - target[ditto]) * rise
    placeholderz[same] = (finbar) + placeholderz[same]
    #dask.delayed(mario)(same,finbar,sima)
    mario(same,finbar,sima)
    #placeholder = dask.compute(*placeholder)
    #placeholder = dask.compute(*placeholder) 

def mario(bbr,b,al): 
   global placeholder
   global zhuge
   global memories
   global counter
   global placeholderz
   for k in range(memories[bbr].size):
           if k >= bbr:
               if al != 0:
                   ryuji = al - 1
                   taiga = k + 1
               else:
                   continue
           else:
               ryuji = al
               taiga = k
           placeholder[bbr][k] = placeholder[bbr][k] + (b * zhuge[ryuji,taiga])
           harm = derivativereLU(zhuge[ryuji,taiga])
           taiping = harm * b * memories[bbr][k]
           placeholderz[bbr] = taiping + placeholderz[taiga]
           counter = counter + 1
           mario(taiga,taiping,ryuji)
           
           
           
           
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

    
    
          
def reconnect(r):               #r is growth rate, number between 0 and 1
  global memories
  fish = np.array([])
  cube = 0
  for x in range(len(memories)):
    for y in range(len(memories[x])):
      if memories[x,y] == None:
        if cube == 0:
          s = x
          e = y
          fish = np.array([[s,e]])
          cube = 1
        else:
          kir = np.array([[x,y]])
          fish = np.append(fish,kir,axis = 0)
  zita = len(fish)
  recet = proportional(zita,r)
  c_plus = memories[0].size
  for x in range(recet):
    ssr = 1
    a = len(fish) -1
    if a > 0:
        recett = np.random.randint(0,a)
        recette = fish[recett][0]
        recettes = fish[recett][1]
        for sss in range(c_plus):
            if memories[recette][sss] != None:
                ssr = ssr + 1
        cp = np.random.randn()
        rrr = cp * ((2/ssr) ** 0.5)           
        memories[recette][recettes] = rrr
        fish = np.delete(fish,recett,axis = 0)
 
def proportional(cd,xs):
  global memories
  r = memories.size
  re = r - cd
  if re != 0:
    rec = re / r
    rece = cd ** ((1 - rec) ** (1 - xs))
  else:
    rece = 0
  rett = rece // 1
  recipe = int(rece)
  return recipe
neurons = np.zeros(shape = 1)
input = np.zeros(shape = 0)
output = np.zeros(shape = 1)
deepness = 6000
target = [[deepness]]
targetindice = [deepness - 1]
memories()
startmemory()
memoriesbias()
fno = len(fullnet)
fullnet = np.zeros(shape = fno)
memoryactivation()
zhuge = np.array([fullnet])
for x in range(deepness - 1):
    memoryactivation()
    zhuge = np.append(zhuge,[fullnet],axis = 0)
counter = 0
ba_zhen_tu(zhuge,target,targetindice)
print(counter)
