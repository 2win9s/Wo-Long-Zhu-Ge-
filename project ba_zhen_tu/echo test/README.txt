parameters saved as XML files made with c++ boost serialisation
coming soon...


For this test the neural network receives a list of random integers between 0 and 9 inclusive, it has to remember one number and then recall it as output some timesteps later.
e.g.

input               output
2,0                   NaN
8,0                   NaN
3,0                   NaN
...                   ...
5,9                   5   //the number that the neural network has to remember, labelled by setting the second input neuron's value to 9
7,0                   NaN
0,0                   NaN
...                   ...
0,1                   5   //we tell the neural network to recall the number by setting the first input neuron to 0 an the second one to 1
