This is empty at the moment
check up backpropagation through time
test the network on long sequence echoing

improve efficiency by using new weights system
3d numpy array, x - neuron, y connections from which neurons, z - the value of the weight
(a list of 2d python arrays)(for speed)
(a list of a list of 2 numpy arrays) (for memory) whichever bottlenecks first
also think of neuron 'growth', have input and output as list of indices of which neurons to read as input/ output
, new neurons that grow are tagged at the end, new input and output neurons are tagged at the end.


how to do the diet function right
The pruning that is associated with learning is known as small-scale axon terminal arbor pruning. Axons extend short axon terminal arbors toward neurons within a target area. Certain terminal arbors are pruned by competition. The selection of the pruned terminal arbors follow the "use it or lose it" principle seen in synaptic plasticity. This means synapses that are frequently used have strong connections while the rarely used synapses are eliminated. So have a basline for (dead/noise) and if the output of that neurons is often under that basline prune it,(achieve this by using the forward pass)


also put a cap on weight size
 if time is enough use ctpyes libruary or just have parts of it in c for efficiency
