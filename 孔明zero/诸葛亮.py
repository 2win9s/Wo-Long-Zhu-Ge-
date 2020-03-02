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
target = np.array([0])

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

    
    
def forget(n):
  global memories
  z = n * -1
  for x in range(len(memories)):
    for y in range(len(memories[x])):
      if memories[x,y] < n:
        if memories[x,y] >= 0:
          memories[x,y] = None
      elif memories[x,y] < 0:
        if memories[x,y] > z:
          memories[x,y] = None
          
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
        else:
          kir = np.array([[x,y]])
          fish = np.append(fish,kir,axis = 0)
  zita = len(fish)
  recet = proportional(zita,r)
  c_plus = memories[0].size
  for x in range(recet):
    ssr = 1
    a = len(fish) -1
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
    rece = cd ** ((1 - re) ** (1 - xs))
  else:
    rece = 0
  rett = rece // 1
  return rett
    
memories()
startmemory()
memoriesbias()
  
