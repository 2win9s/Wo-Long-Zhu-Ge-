FIX THE CODE FOR BACKPROP SOMETHING WENT WRONG

test the network on long sequence echoing

improve efficiency by using new weights system
3d numpy array, x - neuron, y connections from which neurons, z - the value of the weight
(a list of 2d python arrays)(for speed)
(a list of a list of 2 numpy arrays) (for memory) whichever bottlenecks first
also think of neuron 'growth', have input and output as list of indices of which neurons to read as input/ output
, new neurons that grow are tagged at the end, new input and output neurons are tagged at the end.

have dynamic baseline for diet/forget function


also put a cap on weight size
if time is enough use ctpyes libruary or just have parts of it in c for efficiency(write a c module)
