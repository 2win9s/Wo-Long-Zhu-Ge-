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
   placeholder = np.zeros(shape = [len(memories),(len(memories) - 1)]) 
   placeholderz = np.zeros(shape = len(fullnet)) 
   for thing in range(len(target_index)): 
	    sima = target_index[thing] 
	    target = targets[sima] 
	    hardcode(zhuge,target,sima) 

def hardcode(fullnet,target,sima): 
   global output 
   global placeholder 
   global placeholderz 
   global memories 
   faker = sima - 1
   for ditto in range (len(output)): 
    same = (ditto + 1 ) * -1 
    rise = derivativereLU(fullnet[sima][same]) 
    placeholderz[same] = (2 * (fullnet[sima][same] - target[same])) * rise + placeholderz[same] 
    finbar = (2 * (fullnet[sima][same] - target[same])) * rise 
    for x in range(memories[same].size): 
        if ((len(fullnet[sima]) - x ) * -1 ) <= same: 
            if memories[same][x] != None: 
                larry = finbar * memories[same][x] 
                placeholder[same][x] = finbar * fullnet[sima][x] + placeholder[same][x] 
                dice = finbar * memories[same][x] 
                rice = derivativereLU(fullnet[sima][x]) 
                placeholderz[same] = larry * rice + placeholderz[same] 
                mario(x,dice,rice,larry,fullnet,sima) 
        elif memories[same][x] != None:
            if memories[same][x] != None:
                next_three_subjects = x + 1
                larry = finbar * memories[same][x]
                placeholder[same][x] = finbar * fullnet[faker][x + 1] + placeholder[same][x] 
                dice = finbar * memories[same][x] 
                rice = derivativereLU(fullnet[faker][x + 1]) 
                placeholderz[same] = larry * rice + placeholderz[same] 
                mario(next_three_subjects,dice,rice,larry,fullnet,faker) 

def mario(bbr,b,c,d,fin,al): 
   global memories 
   global placeholder       
   global placeholderz 
   kill = 1
   for k in range(len(memories[bbr])): 
        if memories[bbr][k] != None: 
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
            mr = d * memories[bbr][k] 
            placeholder[bbr][k] = b * c * fin[ryuji][taiga] + placeholder[bbr][k] 
            peace = b * c * memories[bbr][k] 
            harm = derivativereLU(fin[ryuji][taiga]) 
            placeholderz[bbr] = mr * harm + placeholderz[bbr] 
            mario(taiga,peace,harm,mr,fin,ryuji)
  
