import numpy as np
import notrandom.py as nr

def statesinit():
  global states
  states = np.array(shape = [0,0]  )
  #k = np.array([])
  #states = np.array([k])
  
def actions(x,y):                             #y is a list of actions e.g. y = ["up","down","left","right","1","2","3","4","5","6","7","8","9"]
  global actions
  actions = np.zeros(shape = x)
  for r in range(y):
    actions[r] = y[r]
 
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
    ii = np.array([x])
    states = np.append(states,ii,axis = 0)
  rh = np.zeros(shape = len(actions))
  for x in range(len(rh)):
    rh[x] = nr.decing(7369130657357778596659,400000010149,195327895579,1,6)
  qtable = np.append(qtable,rh,axis = 0)

  
def resetpf():
    global cc
    global p
    global f
    global rewardlist
    global actionlog
    global past
    global future
    cc = 0
    p = []
    f = []
    past  = None
    future = None
    rewardlist = None
    actionlog = None
    
    
def statemap(xy,tt,action):
  global p
  global f
  global cc
  global past
  global future
  global rewardlist
  global actionlog
  if cc == 0:
    p.append(xy)
    rewardlist = np.array(tt)
    actionlog = np.array(action)
    cc = cc + 1
  if cc == 1:
    if xy != 
      p.append(xy)
      f.append(xy)
      past = np.asarray(p)
      rewardlist = np.append(rewardlist,tt)
      actionlog = np.append(actionlog,action)
      p = None
  if cc == 2:
    past = np.append(past,xy,axis = 0)
    f.append(xy)
    future = np.asarray(f)
    f = None
    rewardlist = np.append(rewardlist,tt)
    actionlog = np.append(actionlog,action)
  if cc > 2:
    past = np.append(past,xy,axis = 0)
    future = np.append(future,xy,axis = 0)
    rewardlist = np.append(rewardlist,tt)
    actionlog = np.append(actionlog,action)
  
  
  
def qupdate(learn,discount):
  global past
  global future
  global states
  global qtable
  global rewardlist
  global actionlog
  for x in range(len(past)-1):
    for y in range(len(states)):
      if past[x] == states[y]:
          for p in range(len(states)):
            if future[x] == states[p]:
                qtable[y,actionlog[x]] = qtable[y,actionlog[x]] + (learn * (reward[x] + (discount * qtable[p,np.argmax(qtable[p])])) - qtable[y,actionlog[x]])
                break
      break
  
------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''def qlearnepsilon():
  global past
  global target
  global qtable
  global actions
  global input
  epsilon = 0.75
  r = np.random.random
  for x in range(len(past)):
    for y in range(len(states)):
      if past[x] == states[y]:
        if epsilon < r:
          target = np.zeros(shape = len(actions))
          target[np.argmax(qtable[y])] = 1
          input = np.copy(past[x])
          nn.fireacrtivation()
          nn.backpropagationpt1()
          nn.updateweights(l)
          nn.updatebias(l)
          epsilon = epsilon * 0.9999
          break
        else:
          epsilon = epsilon * 0.9999
          break
'''
