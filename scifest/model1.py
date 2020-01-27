import numpy as np
import basicML as bc
import pickle
#input list of 18 subjects output 11 different points you cn get as a result pf your grade,[0,12,20,28,37,46,56,66,77,88,100], here we exclude the higher level + 25 points
numofhl = [16,14]
bc.inputgen(18)
bc.hiddenlayersgen()
bc.outputgen(11)
bc.weightsgen()
bc.biasgen()
