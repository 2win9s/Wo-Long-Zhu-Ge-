import numpy as np

def states(x):
  global states
  states = np.append(states,x,axis = 1)
  
def actions(x):
  global actions
  actions = np.zeros(shape = x)
  
def qtable():
  global actions
  global states
  qtable = np.zeros(shape = [shape.size,len(action)])


