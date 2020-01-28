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
def hiddenlayersgen():#if you want 3 hiddenlayers then nmofhl should contain three elements
  global nmofhl       #each element should be the number of neurons you want in that layer
  global hiddenlayers #note:hiddenlayers[x] gets the particular layer and hiddenlayers[x][y] gets the particular neuron
  hiddenlayers = []  
  for x in range(len(nmofhl)):
      hiddenlayers.append(np.zeros(shape = nmofhl[x]))
          
      
  
#creates a 1d numpy array of length x called output
def outputgen(x):
  global output
  output = np.zeros(shape = x)
  
  
def memories():
  global memories
  global input
  global hiddenlayers
  global output
  memories = []
  for x in range(len(hiddenlayers)+2):
    memories.append([])
    if x == 0:
      for z in range(len(hiddenlayers)):
        memories[0].append(np.zeros(shape = [hiddenlayers[z].size,len(input)]))
    elif x <= (len(hiddenlayers)):
      memories[x].append(np.zeros(shape = [len(input),hiddenlayers[x - 1].size]))
      for c in range(x - 1):
        memories[x].append(np.zeros(shape = [hiddenlayers[c].size,hiddenlayers[x - 1].size]))
      for d in range(x,len(hiddenlayers)):
        memories[x].append(np.zeros(shape = [hiddenlayers[d].size,hiddenlayers[x - 1].size]))
    else:
        for k in range(len(hiddenlayers)):
            memories[x].append(np.zeros(shape = [hiddenlayers[k].size,len(output)]))

#creates a list of 1d numpy arrays which acts as the bias
def memoriesbias():
    global memoriesbias
    global memories
    memoriesbias = memories.copy()

def startmemory():
  global memories
  for x in range(memories.size):
    for y in range(memories[x].size):
      for z in range(memories[x][y].size):
        for a in range(memories[x][y][z].size):
          #rrr = np.random.randn()
          #rrr = rrr * ((2/memories[x][y].size)) ** 0.5)
          # or (choose one method)
          #rrr = np.random.random_sample()
          #if rrr < 0.5:
          #rrr = rrr - 1
          memories[x][y][z,a] = rrr

 #firing neurons/ forward pass--------------------------------------------------------------------------   
    
def memories():
  global memories
  global input
  global hiddenlayers
  global output
  memories = []
  for x in range(len(hiddenlayers)+2):
    memories.append([])
    if x == 0:
      for z in range(len(hiddenlayers)):
        memories[0].append(np.zeros(shape = [hiddenlayers[z].size,len(input)]))
    elif x <= (len(hiddenlayers)):
      memories[x].append(np.zeros(shape = [len(input),hiddenlayers[x - 1].size]))
      for c in range(x - 1):
        memories[x].append(np.zeros(shape = [hiddenlayers[c].size,hiddenlayers[x - 1].size]))
      for d in range(x,len(hiddenlayers)):
        memories[x].append(np.zeros(shape = [hiddenlayers[d].size,hiddenlayers[x - 1].size]))
    else:
        for k in range(len(hiddenlayers)):
            memories[x].append(np.zeros(shape = [hiddenlayers[k].size,len(output)]))
#creates a list of 1d numpy arrays which acts as the bias
def memoriesbias():
    global memoriesbias
    global memories
    memoriesbias = memories.copy()

 #firing neurons/ forward pass--------------------------------------------------------------------------   
inputbackup = []
hiddenlayersbackup = []
count = 0


def input_record():
  global input
  global inputbackup
  global count
  if count!= 0 :
    inputbackup.append(np.copy(input))

def memoryactivation():
    global count
    global input
    global hiddenlayers
    global output
    global memories
    global memoriesbias
    global hiddenlayersbackup
    count = count + 1
    if count != 0:
      hiddenlayersbackup.append(hiddenlayers.copy())
    for x in range(len(memories)):
        if x == 0:
            for y in range(len(hiddenlayers)):
                placeholder = np.copy(memories[x][y])
                placeholderz = np.zeros(shape = len(input))
                for z in range(len(memories[x][y])):
                    for a in range(len(memories[x][y][z])):
                        placeholder[z,a] = hiddenlayers[y][z] * memories[x][y][z,a] + memoriesbias[x][y][z,a]
                for p in range(len(placeholderz)):
                    for pl in range(len(placeholder)):
                        placeholderz[p] = placeholderz[p] + placeholder[pl,p]
                for bi in range(len(input)):
                    input[bi] = input[bi] + reLU(placeholderz[bi])
        elif x != (len(memories) - 1) and x != 0:
            for z in range(len(hiddenlayers)):
                placeholder = np.copy(memories[x][z])
                placeholderz = np.zeros(shape = (hiddenlayers[x - 1]).size)
                if z == 0:
                    for k in range(len(memories[x][z])):
                        for kk in range(memories[x][z][k].size):
                            placeholder[k,kk] = input[k] * memories[x][z][k,kk] + memoriesbias[x][z][k,kk]
                    for p in range(len(placeholderz)):
                        for pl in range(len(placeholder)):
                            placeholderz[p] = placeholderz[p] + placeholder[pl,p]
                    for bi in range(hiddenlayers[x - 1].size):
                        hiddenlayers[x - 1][bi] = hiddenlayers[x - 1][bi] + reLU(placeholderz[bi])
                tt = x - 1
                for dee in range(tt):
                    for r in range(len(memories[x][z])):
                        for rr in range(len(memories[x][z][k])):
                            placeholder[r,rr] = hiddenlayers[dee][r] * memories[x][z][r,rr] + memoriesbias[x][z][r,rr]
                    for p in range(len(placeholderz)):
                        for pl in range(len(placeholder)):
                            placeholderz[p] = placeholderz[p] + placeholder[pl,p]
                    for bi in range(hiddenlayers[x - 1].size):
                        hiddenlayers[x - 1][bi] = hiddenlayers[x - 1][bi] + reLU(placeholderz[bi])
                for thing in range(x,len(hiddenlayers)):
                    for r in range(len(memories[x][z])):
                        for rr in range(len(memories[x][z][k])):
                            placeholder[r,rr] = hiddenlayers[thing][r] * memories[x][z][r,rr] + memoriesbias[x][z][r,rr]
                    for p in range(len(placeholderz)):
                        for pl in range(len(placeholder)):
                            placeholderz[p] = placeholderz[p] + placeholder[pl,p]
                    for bi in range(hiddenlayers[x - 1].size):
                        hiddenlayers[x - 1][bi] = hiddenlayers[x - 1][bi] + reLU(placeholderz[bi])
        elif x == (len(memories) - 1):
            for y in range(len(hiddenlayers)):
                placeholder = np.copy(memories[x][y])
                placeholderz = np.zeros(shape = len(output))
                for z in range(len(memories[x][y])):
                    for a in range(len(memories[x][y][z])):
                        placeholder[z,a] = hiddenlayers[y][z] * memories[x][y][z,a] + memoriesbias[x][y][z,a]
            for p in range(len(placeholderz)):
                for pl in range(len(placeholder)):
                    placeholderz[p] = placeholderz[p] + placeholder[pl,p]
            for bi in range(output.size):
                output[bi] = output[bi] + reLU(placeholderz[bi])
        
      
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
    
not finished yet
def backpropagationpt1(realfish):
  global output
  global input
  global target
  global hiddenlayers
  global inputbackup
  global hiddenlayersbackup
  global memories
  global memoriesbias
  placeholder = []
  placeholderz = []
  for x in range(len(memories)):
    placeholder.append([])
    for y in range(memories[x].size):
      placeholder[x].append(0)
      for z in range(memories[x][y].size):
        placeholder[x][y] = np.zeros(shape = [(memories[x][y].size),(memoriesbias[x][y][z].size)])
  for x in range(len(memoriesbias)):
    placeholderz.append([])
    for y in range(memoriesbias[x].size):
      placeholderz[x].append(0)
      for z in range(memoriesbias[x][y].size):
        placeholderz[x][y] = np.zeros(shape = [(memories[x][y].size),(memoriesbias[x][y][z].size)])
  crap = []
  for x in range(realfish):
    crap.append([])
    if x == 0:
     for shadow in range(len(placeholder)):
      shadows = (shadow + 1) * -1
      for kkr in range(placeholder[shadows].size):
        for fire in range(placeholder[shadows][kkr].size):
          for rmb in range(placeholder[shadows][kkr][fire].size):
            crap[x].append(None)
            if shadows == -1:
              if memories[shadows][kkr][fire][rmb] != None:
                placeholder[shadows][kkr][fire][rmb] = hiddenlayers[kkr][fire] * (output[rmb] * (1 - output[rmb])) * (2 * (output[rmb] - target[rmb])) * (1 / len(output)) + placeholder[shadows][kkr][fire][rmb]
                placeholderz[shadows][kkr][fire][rmb] = (output[rmb] * (1 - output[rmb])) * (2 * (output[rmb] - target[rmb])) * (1 / len(output)) + placeholderz[shadows][kkr][fire][rmb]
                crap[x][rmb] = (memories[shadows][kkr][fire][rmb] * (output[rmb] * (1 - output[rmb])) * (2 * (output[rmb] - target[rmb])) * (1 / len(output))
