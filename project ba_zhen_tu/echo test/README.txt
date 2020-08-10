requires openMP and boost.

trained parameters and results coming soon...


For this test the neural network receives a list of random integers between 0 and 9 inclusive as part of the input, it is trained to remember one number and then recall it as output some timesteps later.



input               output

2,0,0               NaN (this number doesn't matter)
8,0,0               NaN
3,0,0               NaN
...                 ...
5,9,0               5   //the number that the neural network needs to remember, labelled by setting the second input neuron's value to 9
7,0,0               NaN
0,0,0               NaN
...                 ...
0,0,9               5   //we tell the neural network to recall the number by setting the third input neuron to 9
