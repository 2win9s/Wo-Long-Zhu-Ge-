import numpy as np


def qtableinit(s,a):            #s is the number of states and a is the number of actions
  qtable = np.zeros(shape = [s,a])
