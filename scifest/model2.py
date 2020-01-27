import numpy as np
import basicML as bc
import pickle
#input list of 18 subjects output 13 different points you cn get as a result of your grade,[0,12,20,28,37,46,56,71,81,91,102,]
numofhl = [16,14]
bc.inputgen(18)
bc.hiddenlayersgen()
bc.outputgen(13)
bc.weightsgen()
bc.biasgen()
