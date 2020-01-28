#this is to randomly split training and test data from a  selection of data, YOU WILL HAVE TO MODIFY THIS FUNCTION IF YOU CARE ABOUT WHAT NAME THE PICKLE FILES HAVE TO BE
import numpy as np
import pickle
subject_list = ["Phy","Acc","Che","Iri","Bus","Pre","Art","Hom","ConSt","Bio","Mus","Geo","Eco","His","Des","AgSci","Fre","Eng"] 
trainsetin = np.array([[None,None,77,56,None,None,None,None,None,100,None,77,None,None,None,None,66,88], [None,None,None,56,100,None,None,None,None,88,None,None,None,77,None,None,77,56], [None,None,None,77,None,None,None,88,None,77,None,88,None,None,None,None,88,88], [None,None,None,56,88,None,66,None,None,100,None,100,None,None,None,None,None,56], [None,None,None,37,56,None,None,None,None,66,77,None,None,None,None,None,None,66], [None,None,None,88,None,None,None,None,None,100,77,88,None,None,None,None,88,88], [None,None,None,88,None,None,66,None,None,66,None,None,None,77,None,None,66,77], [None,None,100,88,None,None,None,None,None,100,None,None,None,88,None,None,100,100], [None,None,None,37,66,70,None,None,None,66,None,None,None,None,None,None,46,56], [None,37,None,20,None,48,None,None,None,66,None,None,None,None,66,None,20,66], [None,None,None,20,88,87,56,46,None,None,None,None,None,46,None,None,None,66], [None,None,None,46,None,59,None,66,None,46,66,None,None,None,None,None,56,56], [None,None,None,46,88,64,None,88,None,66,None,None,None,None,None,88,None,66], [None,None,None,12,46,None,None,None,None,None,None,37,None,56,None,None,0,28], [None,None,None,56,None,None,None,56,None,56,None,None,None,66,None,None,56,77], [None,12,None,0,None,68,None,None,56,None,None,28,None,37,None,None,None,28], [None,0,None,28,None,None,None,None,None,None,None,None,None,77,None,None,46,66], [None,None,None,12,None,78,None,None,77,37,None,None,None,None,56,77,None,37], [None,None,None,46,66,95,None,66,None,56,None,None,None,None,None,None,66,56], [0,0,None,12,None,None,None,None,None,None,None,56,None,None,None,None,12,46], [None,None,0,20,None,None,None,88,None,66,None,None,None,None,None,None,46,46], [None,None,0,20,88,None,None,None,None,77,None,None,56,77,None,None,None,66], [None,None,None,12,37,None,None,None,None,None,None,0,None,46,None,None,0,56], [None,None,None,56,77,None,None,None,None,66,None,None,None,88,56,None,None,77], [None,None,None,12,66,None,None,77,None,37,None,None,None,46,None,None,None,66], [None,None,None,28,37,None,None,None,46,None,None,37,None,None,None,None,None,37], [None,None,None,12,46,61,None,None,None,0,None,None,None,77,None,37,None,37], [None,None,None,56,88,64,None,None,None,77,66,None,None,None,None,None,46,66], [None,0,None,12,None,None,None,None,46,None,None,20,None,37,None,None,None,20], [None,None,0,46,None,None,None,None,None,66,66,None,None,None,None,None,20,77], [None,None,None,56,88,58,None,None,None,77,77,None,None,None,None,None,56,66], [None,None,None,None,None,76,None,None,56,12,None,37,None,None,None,37,None,28], [None,None,None,56,None,61,None,77,None,56,None,46,None,88,None,None,None,77], [None,None,None,0,None,86,66,46,None,None,None,None,None,56,None,0,None,56], [None,None,None,66,None,None,None,66,None,77,None,None,None,None,None,66,56,56], [None,None,None,0,46,None,None,0,None,0,None,None,None,0,None,None,None,46], [None,None,37,12,None,77,None,None,None,None,None,None,None,56,46,None,None,77], [None,12,None,0,56,75,None,None,56,None,None,None,None,None,None,None,None,20], [None,None,88,0,100,85,None,None,None,88,None,None,None,None,None,None,28,66], [37,None,None,46,None,56,None,None,None,None,None,None,None,56,None,None,46,56], [37,None,None,12,None,50,None,None,None,None,None,66,None,77,46,None,None,66], [None,37,None,56,None,61,None,None,None,None,None,56,None,None,None,46,46,88], [0,None,0,12,None,None,None,None,None,66,None,None,None,66,None,None,None,66], [None,None,None,12,46,79,None,None,46,0,None,None,None,37,None,None,None,28], [None,None,None,77,None,None,None,66,None,88,None,None,None,None,None,None,88,77], [None,28,None,None,88,56,None,None,None,None,None,56,None,66,None,None,None,37], [None,None,None,0,None,40,None,56,66,0,None,None,None,46,None,None,None,28], [None,None,None,0,56,51,None,None,None,None,None,None,None,56,None,46,0,46], [None,None,None,12,37,55,None,37,None,0,None,None,None,None,None,None,12,37], [None,None,None,0,None,51,None,46,None,0,None,46,None,56,None,None,None,20], [None,37,None,12,None,None,None,None,None,0,None,None,None,37,0,None,None,28], [77,None,77,88,None,None,None,None,None,100,None,None,None,None,None,None,88,100], [None,None,None,46,88,None,None,None,None,66,None,None,None,77,None,None,56,77], [None,None,None,20,None,None,None,None,88,None,None,56,None,None,None,66,12,66], [None,20,None,28,56,39,None,37,None,None,None,None,None,20,None,None,None,28], [None,None,None,20,56,42,None,None,None,37,None,56,None,None,37,None,None,37], [None,None,None,56,77,94,77,None,None,None,None,None,None,None,None,None,56,77], [None,None,None,20,77,43,None,66,None,None,None,None,None,66,None,None,12,56], [None,None,37,20,None,61,None,None,None,77,None,None,None,77,None,None,46,66], [None,None,None,56,88,98,None,None,88,66,None,None,None,None,None,None,56,66], [46,None,None,56,88,None,None,None,None,66,None,None,None,None,46,None,None,56], [None,None,None,66,None,None,66,None,None,88,None,77,None,88,None,None,None,77], [None,None,None,77,100,66,77,None,None,None,None,None,None,100,None,None,88,77], [None,None,None,77,None,None,None,None,None,None,77,88,None,77,None,None,77,88], [None,None,None,28,77,66,None,88,None,56,None,None,None,None,None,None,28,88], [None,None,None,20,None,None,None,None,None,46,None,46,None,56,None,None,0,66], [None,None,None,20,77,57,None,None,None,66,None,None,None,None,None,66,20,66], [None,None,56,66,None,None,None,None,None,88,None,None,None,77,None,None,46,77], [None,None,None,12,56,None,None,None,None,46,None,56,None,None,None,None,20,46], [None,None,None,0,None,34,None,56,None,0,None,0,None,None,None,None,12,56], [None,None,None,12,56,92,None,None,77,0,None,66,None,None,None,None,None,66], [None,46,None,46,None,None,None,None,None,46,77,None,None,None,None,None,46,56], [None,None,28,46,None,None,None,None,None,66,77,None,None,None,None,None,66,66], [0,None,None,37,None,43,None,None,None,46,None,None,None,66,None,None,56,66], [None,None,None,46,None,None,None,None,None,88,None,77,None,None,None,100,56,77], [None,None,None,46,88,55,None,None,None,46,None,37,None,None,None,56,None,56], [None,None,None,46,88,None,66,None,None,88,None,None,None,None,None,None,None,66], [0,None,0,20,None,53,None,66,None,66,None,None,None,None,None,None,None,66], [None,None,None,20,77,None,None,None,None,0,None,None,None,46,None,None,None,37], [None,None,None,20,88,84,None,77,None,None,None,None,None,None,None,56,46,66], [0,None,None,37,77,None,None,None,None,37,None,None,None,None,66,None,None,66], [None,None,None,66,None,None,66,None,None,None,None,None,None,None,None,56,77,77], [None,None,None,37,88,91,56,46,None,None,None,66,None,None,None,None,None,88], [0,None,None,12,37,None,46,None,None,46,None,None,None,None,None,None,None,56], [None,None,None,12,37,47,None,None,56,12,None,37,None,None,None,None,None,0], [0,None,None,12,None,None,None,None,56,0,None,46,None,None,None,None,None,56], [None,None,None,28,56,None,None,None,None,None,None,56,None,46,None,None,20,77], [None,None,None,0,56,None,None,None,None,46,None,37,None,46,None,None,None,28], [None,None,None,28,66,None,66,37,None,56,None,None,None,None,None,None,None,66], [None,None,None,77,88,None,None,None,None,88,None,None,None,77,None,None,77,77], [None,None,None,28,77,49,None,None,None,None,None,None,None,66,None,46,28,56], [None,None,None,0,None,None,None,None,None,46,None,37,None,37,None,None,20,66], [None,None,0,20,None,54,None,None,None,46,None,None,None,46,None,None,20,66], [56,None,None,None,None,None,None,None,None,100,None,77,None,None,None,None,88,77], [None,None,None,37,56,52,56,46,None,None,None,None,None,0,None,None,None,28], [None,None,None,28,66,None,None,None,None,37,None,None,None,46,None,None,37,56], [None,None,None,88,None,None,None,None,None,88,None,77,None,None,None,None,100,88], [None,37,None,20,56,59,None,None,66,None,None,None,None,66,None,None,None,46], [None,None,None,77,None,None,None,46,None,66,None,None,None,None,None,None,None,77], [None,None,None,56,88,67,None,None,None,66,None,77,None,None,None,66,None,66], [None,None,37,20,None,78,None,66,None,56,None,None,None,None,None,None,12,46], [None,None,None,28,100,64,None,77,None,77,None,66,None,None,None,None,None,88], [None,None,None,66,None,None,None,66,None,77,None,None,None,None,None,66,46,66], [None,None,None,20,None,41,None,None,None,46,None,37,None,None,None,46,20,28], [None,37,None,46,None,None,None,None,None,77,77,None,None,None,None,None,56,66], [None,None,None,20,None,None,None,None,None,56,77,66,None,None,None,None,56,56], [None,None,None,12,46,85,None,46,None,0,None,None,None,None,None,None,20,66], [None,None,None,12,46,48,None,None,46,0,None,None,None,37,None,None,None,28], [None,None,None,0,66,70,None,None,56,0,None,None,None,37,None,None,None,28], [None,28,None,0,66,50,None,None,None,46,None,None,None,46,None,None,None,28], [None,None,None,0,56,None,None,37,None,None,None,None,None,None,None,None,None,46], [None,None,12,20,46,55,None,None,None,56,None,None,None,None,None,None,28,28], [None,None,None,66,None,None,77,None,None,77,77,None,None,None,None,None,88,77], [None,12,None,0,37,45,None,None,None,0,None,None,None,0,None,None,None,28], [None,None,None,66,None,70,None,100,None,88,None,88,None,None,None,None,66,66], [None,46,None,20,88,95,None,None,None,77,None,66,None,None,None,None,None,37], [None,None,None,66,None,None,None,None,None,66,None,None,None,77,56,None,77,77], [None,None,None,20,None,None,None,None,None,46,None,None,None,46,0,None,56,46], [None,None,None,12,None,None,None,77,None,46,None,None,None,77,None,None,56,77], [None,None,None,20,37,None,None,None,None,None,56,0,None,None,None,None,None,46], [None,None,None,46,None,None,None,None,None,56,66,None,None,None,None,46,20,66], [None,None,None,20,None,None,None,None,None,77,None,77,None,None,56,88,None,56], [56,None,88,77,None,None,None,None,None,88,None,None,None,77,None,None,None,88], [None,None,None,46,None,None,None,None,None,66,66,None,None,56,None,None,56,66], [None,None,None,0,56,47,None,66,None,37,None,None,None,None,None,None,None,28], [None,0,None,36,None,None,None,None,None,None,None,None,None,46,None,37,28,66], [None,None,None,37,None,61,None,None,46,None,None,None,None,None,46,None,37,37], [None,37,None,66,None,None,None,None,None,77,None,None,None,46,None,None,None,66], [None,None,66,56,None,None,None,None,None,77,None,None,None,None,None,88,56,56], [None,None,None,56,None,None,None,100,None,66,None,77,None,None,None,None,66,77], [None,None,None,12,56,None,None,None,None,56,None,None,None,56,None,None,56,66], [None,None,None,46,66,None,None,None,None,56,None,56,None,None,46,None,None,66], [None,None,None,66,None,None,None,100,None,88,None,None,None,None,None,88,66,66], [None,None,None,37,None,None,None,66,None,56,None,None,None,None,None,66,20,56], [None,None,None,0,66,None,None,77,None,37,None,56,None,None,None,None,None,46], [None,None,None,46,None,None,None,None,None,46,None,66,None,66,None,None,56,77], [None,46,None,56,None,None,None,100,None,88,None,None,None,None,None,None,66,77], [None,None,None,12,None,None,None,46,None,37,None,46,None,46,None,None,None,56], [None,0,None,56,None,None,None,None,None,46,None,None,None,66,None,None,56,66], [None,None,None,88,88,None,None,None,None,88,None,None,None,77,None,None,88,88], [None,None,None,66,88,None,None,88,None,None,None,88,None,77,None,None,None,77], [None,None,None,28,None,46,None,56,None,46,None,None,None,None,None,None,66,46], [None,None,None,28,None,None,46,None,None,46,None,None,None,None,None,56,56,77], [None,None,None,None,None,None,None,None,46,66,None,None,None,56,None,66,None,66], [None,None,77,56,None,None,None,None,None,56,None,None,None,56,None,None,77,77], [None,None,None,66,88,None,None,None,None,None,77,77,None,None,None,None,77,77], [None,None,None,46,None,None,None,77,None,56,None,None,None,None,None,46,66,56], [None,None,None,46,None,None,None,100,None,77,None,66,None,None,None,None,46,66], [None,None,None,66,77,None,None,None,None,56,None,None,None,None,None,None,66,88], [None,None,None,66,None,None,None,None,None,77,77,None,None,None,66,None,77,77], [None,None,None,56,77,73,None,None,None,None,None,None,None,None,None,56,56,66], [None,None,None,20,66,63,None,None,None,46,None,None,None,None,None,46,None,56], [None,None,None,56,66,None,None,None,None,56,None,66,None,None,None,None,56,77], [None,None,None,12,56,69,46,None,None,None,None,None,None,None,None,37,None,77,], [None,None,None,0,None,None,None,None,37,46,None,None,None,77,None,None,20,46], [None,None,None,0,None,None,None,None,None,46,None,56,None,37,None,None,None,56], [None,46,None,66,None,None,None,88,None,None,None,None,None,None,None,66,77,77], [None,None,None,56,None,None,None,46,None,56,77,None,None,None,None,None,56,66], [None,None,None,77,66,None,None,None,None,66,None,77,None,None,None,None,77,56], [None,37,None,56,66,69,None,None,None,None,None,None,None,None,None,46,None,66], [None,None,None,12,None,None,None,None,None,56,None,56,None,56,None,37,None,77], [None,None,66,37,None,None,None,None,None,77,None,77,None,None,None,None,66,66], [None,None,None,77,77,None,None,None,None,77,77,None,None,None,None,None,77,100], [None,None,56,37,None,None,66,None,None,88,None,None,None,None,None,None,46,88], [None,None,None,56,66,None,None,None,None,56,None,66,None,None,None,None,59,66], [77,None,77,66,None,None,None,None,None,77,None,None,None,77,None,None,None,88], [None,None,None,20,None,None,None,None,None,56,None,66,None,56,None,None,37,77], [None,None,None,56,77,None,None,None,None,77,None,None,None,None,None,77,46,66], [None,None,None,20,46,76,None,None,None,37,None,None,None,None,37,None,None,56], [None,None,None,20,None,61,None,None,37,None,None,46,None,None,None,37,None,46], [None,None,None,20,None,None,None,None,None,66,None,None,None,66,56,66,None,66], [None,None,None,56,66,None,None,None,None,37,None,None,None,77,None,None,37,77], [None,None,None,28,None,None,None,77,None,None,56,None,None,None,None,47,None,56], [None,None,None,0,0,34,None,0,None,12,None,None,None,None,None,None,None,12], [37,None,None,20,None,None,None,None,None,None,56,None,None,None,56,None,46,66], [None,None,None,0,56,None,None,None,None,56,None,None,None,46,None,None,37,46], [None,None,56,66,None,None,None,66,None,66,None,None,None,None,None,None,77,66], [None,None,66,88,None,None,None,None,None,None,None,None,None,88,100,None,77,88], [None,None,None,28,None,None,None,None,None,37,None,46,None,37,None,None,20,77], [None,None,None,66,None,None,None,None,None,None,77,88,None,None,None,88,66,77], [None,0,None,0,0,64,None,None,None,0,None,None,None,None,None,None,None,20], [None,None,None,0,None,None,None,None,0,None,None,None,None,12,None,None,None,20], [None,None,None,37,None,None,None,None,None,56,66,None,None,66,None,None,56,77], [None,None,None,0,56,None,None,77,None,None,None,None,None,None,None,37,56,37], [None,0,None,0,None,None,None,56,None,None,None,46,None,None,None,None,28,56], [100,None,77,88,None,None,None,None,None,None,None,None,None,None,46,None,88,88], [None,None,None,12,None,None,None,0,0,None,None,12,None,12,None,None,None,20], [46,None,37,0,None,None,None,None,None,None,None,37,None,20,None,None,None,0], [None,None,None,28,None,71,56,None,None,56,None,None,None,None,None,46,None,66], [None,None,None,12,None,70,None,None,None,56,None,None,None,37,None,37,None,66], [None,None,None,12,37,None,None,56,None,0,None,None,None,0,None,None,None,28], [100,None,None,37,None,None,None,None,None,77,None,None,None,None,66,100,None,66], [66,None,None,37,88,None,None,None,None,None,None,77,None,None,None,None,56,77], [None,37,None,56,None,None,None,None,None,77,None,None,None,46,None,None,46,46], [None,None,None,37,None,None,None,None,None,0,None,56,None,None,None,66,28,66], [0,None,None,12,None,None,None,None,None,46,None,None,None,None,77,None,56,77], [None,None,None,56,77,None,None,None,None,None,None,None,None,56,None,56,77,88], [None,None,None,20,56,64,None,66,None,None,None,46,None,None,None,None,None,56], [None,None,None,12,None,None,None,None,0,56,None,46,None,0,None,None,None,20], [None,None,None,0,None,None,None,46,None,0,None,46,None,37,None,None,None,28], [None,None,None,46,56,None,None,None,None,46,None,None,None,66,None,None,46,46], [None,None,None,20,37,None,None,None,None,20,None,46,None,None,None,None,12,37], [None,20,None,12,37,None,None,None,None,20,None,None,None,None,None,66,None,28], [None,None,None,77,None,None,None,100,None,88,77,None,None,None,None,None,88,77], [None,None,None,20,66,None,None,None,None,37,None,None,None,46,None,None,37,56], [None,None,None,12,None,None,None,None,None,56,56,66,None,66,None,None,None,66], [None,None,None,56,100,None,None,88,None,None,None,None,None,None,None,37,56,77], [None,None,None,12,37,None,None,None,None,0,None,20,None,56,None,None,None,28], [None,None,None,0,None,None,None,None,None,66,None,77,None,None,None,66,56,56], [None,0,None,0,0,36,None,None,None,0,None,None,None,None,None,None,None,0], [None,None,None,46,None,None,None,None,None,37,None,None,None,56,None,46,46,88], [0,None,None,0,None,None,None,None,0,46,None,37,None,None,None,None,None,46], [None,None,None,66,77,None,None,None,None,56,None,66,None,None,None,None,46,56], [None,None,None,20,None,None,None,66,None,None,66,66,None,46,None,None,None,37], [None,None,None,12,37,None,None,37,None,0,None,None,None,0,None,None,None,20], [None,None,None,77,None,None,None,100,None,100,None,None,None,None,None,100,66,88], [None,None,None,46,None,None,None,77,None,46,None,56,None,None,None,None,46,56], [None,37,28,46,None,None,None,77,None,None,None,None,None,None,None,None,66,77], [None,None,None,28,None,None,None,None,None,46,None,56,None,None,None,46,56,56], [None,None,None,12,56,None,None,None,None,56,None,None,None,56,None,37,None,46], [None,None,None,0,None,None,None,None,37,None,None,20,None,None,None,0,None,12], [None,None,None,46,None,None,None,None,None,66,66,None,None,None,None,None,56,66], [None,None,None,0,12,None,None,None,None,None,56,12,None,0,None,None,None,20], [56,None,None,46,None,None,None,None,None,66,None,28,None,None,None,None,77,56], [None,None,77,88,None,None,None,None,None,88,None,None,None,None,None,100,77,88], [None,None,None,12,56,None,None,None,None,56,None,None,None,56,None,None,28,66], [None,None,None,46,46,None,None,66,None,None,None,46,None,None,None,None,12,56], [None,None,77,88,None,None,None,None,None,88,None,None,None,56,None,None,88,100], [None,None,None,46,46,None,None,None,None,12,None,37,None,None,None,None,12,46], [None,37,None,None,37,None,None,None,None,37,None,None,None,None,None,None,12,46], [100,None,None,88,None,None,None,None,None,100,None,None,None,88,77,None,None,88], [None,None,None,12,46,69,None,None,0,37,None,None,None,None,None,None,None,20], [None,None,None,46,77,None,None,None,None,46,None,None,None,77,None,None,56,77], [None,None,None,0,46,None,None,77,None,37,None,None,None,None,None,None,37,46], [None,None,0,56,46,None,None,None,None,46,None,None,None,None,None,None,56,56], [None,None,None,20,None,51,None,46,None,None,None,None,None,None,None,0,12,56], [None,None,None,0,None,None,None,None,None,46,None,66,None,46,46,None,None,56], [100,None,None,0,77,None,None,None,None,56,None,None,None,None,None,None,56,66], [None,None,None,0,37,56,None,46,None,37,None,None,None,None,None,None,None,12], [None,None,None,77,None,None,None,100,None,77,None,66,None,None,None,None,77,77], [None,None,None,None,None,None,None,None,None,66,None,None,None,0,77,None,66,56], [None,37,None,56,None,None,None,None,None,37,None,66,None,None,None,None,46,56], [None,None,None,28,None,None,None,None,None,46,None,None,None,56,None,37,46,66], [None,None,None,20,46,None,None,None,None,46,None,66,None,None,None,None,20,56], [None,None,None,20,None,None,0,37,None,37,None,None,None,None,None,37,None,0], [None,None,None,12,None,None,46,46,None,None,None,None,None,46,None,None,None,0], [None,None,None,12,None,None,None,None,0,37,None,None,None,56,None,46,None,28], [None,None,None,56,66,None,None,None,None,56,None,None,None,None,46,56,None,66], [None,None,None,28,88,None,None,None,None,37,None,None,None,56,None,None,56,77]])
trainsetout = np.array([77,66,12,77,46,66,46,100,46,56,0,0,12,0,12,0,28,28,0,37,37,46,12,37,20,12,0,46,0,37,56,0,37,0,37,0,46,0,12,37,37,46,46,0,46,12,0,20,0,0,12,88,46,46,0,12,0,0,46,46,46,56,66,56,0,0,12,56,46,0,12,46,37,0,46,37,0,28,0,12,46,56,20,0,0,37,0,0,0,12,12,0,0,77,0,0,20,0,56,28,12,12,56,12,46,46,0,0,0,0,0,0,66,12,56,28,88,20,12,20,20,77,66,12,28,28,56,28,37,46,12,28,37,20,28,37,46,0,46,77,28,28,12,12,46,56,46,28,28,46,56,20,12,12,12,20,66,28,56,28,0,46,56,56,12,56,12,46,28,12,37,37,12,0,37,0,37,88,0,77,20,0,12,12,46,77,20,20,12,28,0,88,66,20,20,56,20,0,0,0,12,0,20,28,12,12,20,0,66,0,20,12,28,28,0,66,20,37,12,20,0,56,12,56,77,28,0,56,12,12,88,20,28,20,20,12,66,46,20,46,46,37,0,37,12,0,12,46,28]) 
trainsetoutz = np.array([102,91,12,102,71,91,71,125,71,81,0,0,12,0,12,0,28,28,25,62,62,71,12,62,20,12,0,71,0,62,81,0,62,0,62,0,71,0,12,62,62,71,71,0,71,12,0,20,0,0,12,113,71,71,0,12,0,0,71,71,71,81,91,81,0,0,12,81,71,0,12,71,62,0,71,62,0,28,0,12,71,81,20,0,0,62,0,0,0,12,12,0,0,102,0,0,20,25,81,28,12,12,81,12,71,71,0,0,0,0,0,0,91,12,81,28,113,20,12,20,20,102,91,12,28,28,81,28,37,46,12,28,37,20,28,62,46,0,71,102,28,28,12,12,71,81,71,28,28,71,81,20,12,12,12,20,91,28,81,28,0,71,81,81,12,81,12,71,28,12,62,37,12,0,62,25,62,113,0,102,20,0,12,12,46,102,20,20,12,28,0,113,91,20,20,81,20,0,0,0,12,0,20,28,12,12,20,0,91,0,20,12,28,28,0,91,20,62,12,20,0,81,12,81,102,28,25,81,12,12,113,20,28,20,20,12,91,71,20,71,46,37,0,37,12,0,12,71,28]) 
trainsetinz = np.copy(trainsetin)
def testdataseparate(k,y,z): #k is number of sets of  data,y is list of  input,z is list of  outputs, REMEMBER k CANNOT BE LARGER THEN len(y)
    listindice = []
    while len(listindice) <= (k - 1):
        fish = np.random.randint(0,len(y))
        if len(listindice) == 0:
            listindice.append(fish)
        rrr = 0
        for x in range(len(listindice)):
            if listindice[x] == fish:
                rrr = 1
                break
        if rrr != 1:
            listindice.append(fish)
            
    for ssr in range(len(listindice)):
        if ssr == 0:
            umm = listindice[ssr]
            ur = np.copy([y[umm]])
            mr = np.copy(z[umm])
        else:
            t = listindice[ssr]
            ur = np.append(ur,[y[t]],axis = 0)
            mr = np.append(mr,z[t])
    y = np.delete(y,listindice,axis = 0)  
    z = np.delete(z,listindice,axis = 0)
    with open("traininput.p","wb") as aa:
        pickle.dump(y,aa)
    #with open("traininput.p","rb") as ab:
        #traininput = pickle.load(ab)    #this makes traininput equal to the stuff in the picklefile
    with open("trainoutput.p","wb") as ba:
        pickle.dump(z,ba)
    #with open("trainoutput.p","rb") as bb:
        #trainoutput = pickle.load(bb)
    with open("testinput.p","wb") as ca:
        pickle.dump(ur,ca)
    #with open("testinput.p","rb") as cb:
        #testinput = pickle.load(cb)
    with open("testoutput.p","wb") as da:
        pickle.dump(mr,da)
    #with open("testoutput.p","rb") as db:
        #testoutput = pickle.load(db)
