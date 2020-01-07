import numpy as np
#shortened decriptions and tags i will be using for parameters

#      meme (minimize for efficiency,maximize for effectiveness)

#      anti-meme (maximize for efficiency,minimize for effectiveness), should come up rarely

#      prime (must be a prime number)

#      note (do not ignore what comes after it)

#      mtm  (more the merrier;less significant figures more efficiency,more significant figures more effectiveness)






#useful basic math operators

# * means multiply % means modulo ** means to the power of +,- the same / division

# // floor division removes decimal points from answer,by always lowering the value to the next whole number e.g. -4.15->-5 , 3.87->3

# == equality check e.g. a==a+1 is not true and a==a is true,!= inequality check,a!=a+1 is true and a!= is not true similiar to <>

# > if greater than then true a+1>a is true a>a+! untrue, < if less than true a+1<a untrue a<a+1 tue,>= greater and equals true,<= less than equals true

# = assigns value on right to the variable on the left e.g. b = a + 1 makes b a + 1

# += , c += a is the same as c = c + a

# -= , c -= a is the same as c = c - a, *= , c *= a is the same as c = c * a, /=, %=,**=,//=, you get the idea

# order of sums: brakets,exponents,multiply divide modular arithmatic floor division,addition subraction,comparison operators e.g. <,equality operators e.g. ==, finally the funky shortened sum things e.g. +=





#math functions




# simple rng,note: can only produce positive random numbers 
def random(x,m):                                   #x is the seed can be any pseudo random number,dice the time in milliseconds etc.(recommend a prime number),m is maximum value from rng(recommend a prime number)
#parameters of rng
  global rngout
  a = (x * rngout) + (x  * m)
  rngout = a % m
                                                  #rngout is the random number produced(note  r is a second seed)


#reLU, most efficient activation algorithm for hidden layers 
def reLU(x):
  global reLUout
  if x > 0:
    reLUout = x
  else:
    a = 0.01                                      #zero for normal reLU, small number for leaky reLU,keep it as a learned parameter for Para Relu(effective not efficient,evolution may a good way to implement if there are  other parameterss that would also evolve)
    reLUout = x*a                                 #you can change this formula if you want something different where x is negative, you can study how different functions here affect things



def eulerno(x):                                    #note:run this function once to give a value to euler's number note:both e and x are meme
  global eulern                                   #euler's number or e
  eulern = (1+1/x)**x

def sigmoid(x):                                    #good output function for classsification problems with multiple outputs)
  global eulern                                   #requires euler's number
  global sigout 
  var = 0.1                                       #anti-meme, can be adjusted with learning or evolution,note: var > 1 is risky and possibly very ineffective,can ruin your entire network
  sigout = 1 / ( 1 + ( eulern ** ( var * x * (-1) ) ) )





#neural network set up




#creates an input layer
def inputgen(x):
  global input
  input = np.zeros(shape = x)
    

def inputwrite(x,y):                               #assigns a value y to element x of the input,note:lists start at 0 so first element is 0 2nd 1 one etc.
  global input
  input[x] = y


#hidden layers,i will store these as a list of lists, i know that it is not efficient but if translated to arrays in c there will be no problem    
def numofnugen(list_that_numofnu_should_be)
  numofnu = list_that_numofnu_should_be

def hiddenlayergen():
  global numofnu
  global hiddenlayers
  hiddenlayers = numpy.array([])                                           #create a list of the number of neurons in each layer here
  for x in range(len(numofnu)):
    hiddenlayers = numpy.append(hiddenlayers,[],axis = 1)
    for y in range(numofnu[x]):
      hiddenlayers[x] = numpy.append(hiddenlayers[x],0,axis = 2)
      
  



#creates an output layer

def outputgen(x):
  global output
  output = numpy.zeros(shape = x)



def weightgen():                                     #creates a list of list of weights aka fustration 
  global weights
  global hiddenlayers
  global input                                    #the number of weights depends on inputs and hidden layers
  weights = numpy.array([])
  k = len(hiddenlayers)+1                         #there is one extra layer of weights because of output
  for amount in range(k):
    weights = numpy.append(weights,[],axis = 1)                            #creates the number of sets of weights
  for x in range(len(input)):           # the special set of weights from input to first hidden layer
    weights[0] = numpy.append(weights[0],0,axis=2)                         #adds a row of weights for each input
    for y in range(hiddenlayers[0].size):
      weights[0][x] = numpy.append(weights[0][x],0,axis=3)       #adds weights to each row based on hidden layer
  for x in range(len(hiddenlayers)):
     y = x + 1
     if y < len(hiddenlayers):
      for z in range(len(hiddenlayers[x])):         
        weights[y].append([])                     #each matrix of weights has a number of rows
        for a in range(len(hiddenlayers[y])):
           weights[y][z].append(0)                #each row has a number of weights
     else:
       for z in range(len(hiddenlayers[-1])): 
        weights[-1].append([])                    #special case for output 
        global output
        for a in range(len(output)):
           weights[-1][z].append(0)


       

def biasgen():                                       #generates the list of biases
  global bias
  global hiddenlayers
  global output
  bias = hiddenlayers.copy()
  bias.append(output)



 
 #firing section



def squish():                                        #so this squishes a 2-d matrix of number from multiplying the input to weights to get the next layer
  global placeholder
  global placeholderz
  placeholderz.clear()
  for x in range(len(placeholder[0])):
    placeholderz.append(0)
    for y in range(len(placeholder)):
      placeholderz[x] = placeholder[y][x] + placeholderz[x]
      
def clear():                                         #resets the placeholder lists
  global placeholder
  global placeholderz
  placeholder.clear()
  placeholderz.clear()
def set():                                           #sets up the placeholders
  global placeholder
  global placeholderz
  
  placeholder.append([])



def reLUed():                                        #uses the reLU function on placeholderz
  for ez in range(len(placeholderz)):
        global reLUout
        reLU(placeholderz[ez])
        placeholderz[ez] = reLUout

def sigs():
   for ez in range(len(placeholderz)):
        global sigout
        sigmoid(placeholderz[ez])
        placeholderz[ez] = sigout


def fireactivation():                                #this is the network firing up its neurons
  global input
  global hiddenlayers
  global weights
  global bias
  global placeholder
  global placeholderz
  global output
  placeholderz = []
  placeholder = []
  clear()
  for x in range(len(input)):                        #going from input to hidden layers
    set()
    for weighted in range(len(hiddenlayers[0])):
      placeholder[x].append(input[x] * weights[0][x][weighted])
  squish()
  for x in range(len(placeholderz)):
    placeholderz[x] = placeholderz[x] + bias[0][x]
  reLUed()                                           #activation function
  hiddenlayers[0] = placeholderz.copy()
  for x in range(len(hiddenlayers)):                 #going through the hidden layers
    clear()
    y = x + 1
    if y < len(hiddenlayers):
      for z in range(len(hiddenlayers[x])):
        set()
        for k in range(len(hiddenlayers[y])):
            placeholder[z].append(hiddenlayers[x][z] * weights[y][z][k])
      squish()
      for r in range(len(placeholderz)):
          placeholderz[r] = placeholderz[r] + bias[y][r]
      reLUed()                                         #activation function
      hiddenlayers[y] = placeholderz.copy()
      clear()
    else:                                              #going from final hidden layer to output
      clear()
      for z in range(len(hiddenlayers[-1])):
         set()
         for weighted in range(len(output)):
            placeholder[z].append(hiddenlayers[-1][z] * weights[-1][z][weighted])
      squish()
      for x in range(len(placeholderz)):
        placeholderz[x] = placeholderz[x] + bias[-1][x]
      sigs()                                            #activation function
      output = placeholderz.copy()
      clear()
        
        




                                       #give a target output              

for x in range(len(output)):
  global target
  target = []
  target.append(0)

def targetwrite(x,y):
  global target
  target[x]= y

def msecostgen():
  global output
  global target
  global cost
  global costly
  cost = 0
  costly = []
  for x in range(len(output)):
    costly.append((output[x] - target[x]) ** 2)
  for y in range(len(costly)):
    cost = cost + costly[x]
  k  = len(output)  
  cost  =  cost / k
  
  


def derivativereLU(x):                                         #x is the value of the input to the reLU function 
  global dereLU
  if x > 0:
    dereLU = 1
  else:
    a = 0.01                                    
    dereLU = a


def backpropagation():
  global weights
  global output
  global hiddenlayers
  global bias
  global ouput
  global target
  global input
  global dereLU
  clear()
  global placeholder
  global placeholderz
  placeholder = []
  placeholderz  = []
  for x in range(len(bias)):
    placeholderz.append([])
    for y in range(len(bias[x])):
      placeholderz[x].append(0)
  for x in range(len(weights)):
    placeholder.append([])
    for y in range(len(weights[x])):
      placeholder[x].append([])
      for z in range(len(weights[x][y])):
        placeholder[x][y].append(0)
  
  for x in range(len(output)):
    crap = 0
    placeholderz[-1][x] = (output[x]*(1 - output[x])) * (2 * (output[x] - target[x]))* (1 / len(output)) + placeholderz[-1][x]
    
    #bit to be copied & pasted for each layer of bias starts here 
    
    for y in range(len (weights[-1][x])):
      crap = weights[-1][x][y] * (2 * (output[x] - target[x])) * (2 * (output[x] - target[x]))* (1 / len(output))
      for z in range(len(placeholderz[-2])):
        craap = 0
        derivativereLU(hiddenlayers[-1][z])
        placeholderz[-2][z] = dereLU * crap + placeholderz[-2][z]
    #bit to be copied & pasted for each layer of bias ends here
        
        for a in range(len(weights[-2][z])):
          craap = weights[-2][z][a] * dereLU * crap
          for b in range(len(placeholderz[-3])):
            craaap = 0
            derivativereLU(hiddenlayers[-2][b])
            placeholderz[-3][b] = dereLU * craap + placeholderz[-3][b]
                
            for c in range(len(weights[-3][b])):
                craaap = weights[-3][b][c] * dereLU * craap
                for d in range(len(placeholderz[0])):
                    craaaap = 0
                    derivativereLU(hiddenlayers[-3][d])
                    placeholderz[-4][d] = dereLU * craaap + placeholderz[-4][d]
  
  
  for x in range(len(hiddenlayers[-1])):
      for y in range(len(placeholder[-1][x])):
        shit = 0
        placeholder[-1][x][y] = hiddenlayers[-1][x] * (output[y]*(1 - output[y])) * (2 * (output[y] - target[y]))* (1 / len(output)) + placeholder[-1][x][y] 
        shit = weights[-1][x][y] * (output[y]*(1 - output[y])) * (2 * (output[x] - target[x]))* (1 / len(output))
        
        #bit to be copied & pasted for each layer of weights starts here
        
        for z in range(len(hiddenlayers[-2])):  
          for a in range(len(placeholder[-2][z])):
            shiit = 0
            derivativereLU(hiddenlayers[-1][a])
            placeholder[-2][z][a] = hiddenlayers[-2][z] * dereLU * shit + placeholder[-2][z][a]
            shiit = dereLU * weights[-2][z][a] * shit
            
            for b in range(len(hiddenlayers[-3])):
              for c in range(len(placeholder[-3][b])):
                shiiit = 0
                derivativereLU(hiddenlayers[-2][c])
                placeholder[-3][b][c] = hiddenlayers[-3][b] * dereLU * shiit + placeholder[-3][b][c]
                shiiit = dereLU * weights[-3][b][c]  * shiit
                
                for d in range(len(input)):
                    for e in range(len(placeholder[-4][d])):
                        shiiiit = 0
                        derivativereLU(hiddenlayers[-3][e])
                        placeholder[-4][d][e] = input[d] * dereLU * shiiit + placeholder[-4][d][e]
                        shiiiit = dereLU * weights[-4][d][e]  * shiiit
        
 #use as template copy and paste a feew times for multiple layers of weights and bises remember to switch around variables and schtuff like that
#"The best that most of us can hope to achieve in physics is simply to misunderstand at a deeper level." -- Wolfgang Pauli
#every thing in useful things & tools works !!!!!!!
