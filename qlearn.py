import numpy as np
import notrandom.py as nr

def statesinit():
  global states
  states = np.array(shape = [0,0]  )
  
  
def actions(x,y):                             #y is a list of actions e.g. y = ["up","down","left","right","1","2","3","4","5","6","7","8","9"]
  global actions
  actions = np.zeros(shape = x)
  for r in range(y):
    actions[r] = y[r]

def qtableinit():
  global qtable
  
def qtableaddstate(x):
  global states
  global actions
  global qtable
  r = 0
  for k in range(len(states)):
    if x == states[k]:
      r = r + 1
    else:
      r = r + 0
  if r == 0:
    states = np.append(states,x,axis = 1)
  rh = np.zeros(shape = len(actions))
  for x in range(len(rh)):
    rh[x] = nr.decing(7369130657357778596659,400000010149,195327895579,1,6)
  qtable = np.append(qtable,rh,axis = 1)

def pfstates():
  global p
  global f
  global past
  global future
  past = np.asarray(p)
  future = np.asarray(f)

  
def resetpf():
    global cc
    global p
    global f
    cc = 0
    p = []
    f = []
    

def statemap(xy):
  global p
  global f
  global cc
  if cc == 0:
    p.append(xy)
  if cc == 1:
    p.append(xy)
    f.append(xy)
  if cc == 2:
    p.append(xy)
    f.append(xy)
  cc = cc + 1
    
  
  
  

