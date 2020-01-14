import numpy as np

def statesinit():
  global states
  states = np.array([])
def actions(x):                             #y is a list of actions e.g. y = ["up","down","left","right","1","2","3","4","5","6","7","8","9"]
  global actions
  actions = np.zeros(shape = x)
 
def qtableaddstate(x):
  global states
  global actions
  global qtable
  r = 0
  for k in range(len(states)):
    if np.array_equal(x,states[k]):
      r = r + 1
    else:
      r = r + 0
  if r == 0:
      if len(states) == 0:
          states = np.array([x])
          rh = np.zeros(shape = len(actions))
          rhn = np.array([rh])
          for x in range(len(rh)):
            rhn[0][x] = np.random.random_sample()
          qtable = rhn
      else:  
          ii = np.array([x])
          states = np.append(states,ii,axis = 0)
          rh = np.zeros(shape = len(actions))
          rhn = np.array([rh])
          for x in range(len(rh)):
            rhn[0][x] = np.random.random_sample()
          qtable = np.append(qtable,rhn,axis = 0)

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
    
def statemap(xy,rw,action):
  global p
  global f
  global cc
  global past
  global future
  global rewardlist
  global actionlog
  if cc > 2:
    jj = np.array([xy]) 
    past = np.append(past,jj,axis = 0)
    future = np.append(future,jj,axis = 0)
    rewardlist = np.append(rewardlist,rw)
    actionlog = np.append(actionlog,action)
    cc = cc + 1
  if cc == 2:
    jj = np.array([xy])  
    past = np.append(past,jj,axis = 0)
    f.append(xy)
    future = np.asarray(f)
    f = None
    rewardlist = np.append(rewardlist,rw)
    actionlog = np.append(actionlog,action)
    cc = cc + 1	
  if cc == 1:
      p.append(xy)
      f.append(xy)
      past = np.asarray(p)
      rewardlist = np.append(rewardlist,rw)
      actionlog = np.append(actionlog,action)
      p = None
      cc = cc + 1
  if cc == 0:
    p.append(xy)
    rewardlist = np.array(rw)
    actionlog = np.array(action)
    cc = cc + 1
  
def qupdate(learn,discount):
  global past
  global future
  global states
  global qtable
  global rewardlist
  global actionlog
  for x in range(len(past)-1):
    for y in range(len(states)):
      if np.array_equal(past[x],states[y]):
          for p in range(len(states)):
            if np.array_equal(future[x],states[p]):
                qtable[y,actionlog[x]] = qtable[y,actionlog[x]] + (learn * (rewardlist[x] + (discount * qtable[p,np.argmax(qtable[p])])) - qtable[y,actionlog[x]])
                print(qtable[y,actionlog[x]])
                break
          break
  
------------------------------------------------------------------------------------------------------------------------------------------------------------------
''' stuff to put in the main thing
epsilon = 0.667
def qlearnepsilon():
  global past
  global future
  global target
  global qtable
  global actions
  global input
  global epsilon
  r = np.random.random
  for x in range(len(past)-1):
    for y in range(len(states)):
      if past[x] == states[y]:
        if epsilon < r:
          target = np.zeros(shape = len(actions))
          target[np.argmax(qtable[y])] = 1
          input = np.append(past[x],future[x])
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
