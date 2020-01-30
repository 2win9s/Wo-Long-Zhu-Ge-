import numpy as np
import time as tm
print ("welcome to tic tac toe")

GameBoard = np.array([[0,0,0],[0,0,0],[0,0,0]])

#print (GameBoard)
#print (GameBoard[0][1])

m1 = int(input())

P1 = " "
P2 = " "
P3 = " "
P4 = " "
P5 = " "
P6 = " "
P7 = " "
P8 = " "
P9 = " "


def DispBoard():
  print ("   ¦   ¦   ")
  print ("-----------")
  print ("   ¦   ¦   ")
  print ("-----------")  
  print ("   ¦   ¦   ")

if m1 is 1:
  if GameBoard[0][0] == 0:
    GameBoard[0,0]=1
    DispBoard()
elif m1 is 2:
  print ("slot 2")
elif m1 is 3:
  print ("slot 3")
elif m1 is 4:
  print ("slot 4")
elif m1 is 5:
  print ("slot 5")  
elif m1 is 6:
  print ("slot 6") 
elif m1 is 5:
  print ("slot 7") 
elif m1 is 5:
  print ("slot 8") 
elif m1 is 9:
  print ("slot 9") 
else:
  print ("false")

'''for (m1 is 1):
  print ("true")
  #tm.sleep(1)
else:
  print ("false")
'''
