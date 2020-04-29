NOTE TO SELF/TO DO LIST UPDATE INIT, DEBUG TRAIN, MAKE MODULE



SciFest@College AIT 2020 project by Kacper Gibas, Sam Kierans and Billy Yan.

apologies we do not know how to put in citations so we just stuck the links in instead. sorry about that


Retaining information through arbitrary timesteps with 'simple' RNN?
Kacper Gibas, Sam Kierans and Billy Yan.



A basic RNN(https://crl.ucsd.edu/~elman/Papers/fsit.pdf) achieves memory by receiving 2 sets of inputs;new information input and previous hidden state(context units) and returning 2 sets of outputs the output and the new hidden state.

LSTM(https://www.bioinf.jku.at/publications/older/2604.pdf) perhaps the most successful type of RNN, uses hidden states and gates to retain information for arbitrary timesteps. This allows LSTMs ,when combined with other techniques such as transformers(https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) become state of the art in practically all areas of a.i. research where it is applicable.    

Is a hidden state essential for retaining information for many timesteps?

Biological neural networks (brains) don't seem to use one; they probably don't send of its information to a separate entity in order to wipe or reset itself, I think we can all agree that the information stays in the brain and doesn't get wiped. If the brain is able to work by keeping all the information in the brain, why don't we try do the same thing with artificial neural networks?

What we aim to do is retain information for many timesteps without the use hidden states, to see if anything interesting happens.
How could a network retain information without a hidden state?
The obvious method is to not reset the neural network every timestep and just do all the calculations on top of the 'current value' of the neurons.
i.e. for a particular neuron  new_value = activation_function(current_value + Σinputs ⊙ weights  + bias)
current_value is the value of the neuron from the previous timestep, new_value is the new value of the neuron for this timestep
by (Σinputs ⊙ weights) we mean the sum of the elements of the Hadamard product of the input vector/tensor and weight vector/tensor for a particular neuron, and bias means the bias of a particular neuron.

Can a network that works this way hold information for arbitrary timesteps?
let's assume e=π=3 (insert other ridiculous and covenient approximation/assumption of choice) and do some probably inaccurate speculating.

One way to try and prove that it is possible to hold information for arbitrary timesteps is to show for a particular neuron new_value can be equal to current_value. If new_value can be equal to current_value for given criteria as long as arbitrarily many timesteps fit that criteria then that information can theoretically stay in the network indefinitely i.e. holding information for arbitrary timesteps.

First we ignore input,weights and bias considering just the activation function,we need: new_value = activation_function(current_value), sadly we are forced to use non-linear activation functions in most neural networks, or else the entire network becomes a linear regression model; a useful tool but usually not what most recurrent neural networks are trying to model. Non-linear activation functions would theoritically change the input at least bit by bit through every timestep until it becomes useless as a memory. So mission failed already? 

No, we are saved by the saviour of deep learning: reLU!(https://ui.adsabs.harvard.edu/abs/2000Natur.405..947H/abstract), 
for New_value = reLU(Current_value) to be true we just need current_value >= 0 becuase reLu is linear as long as current_value >= 0.

Next, we need current_value = current_value + Σinputs ⊙ weights + bias, so lets assume bias = 0, then we need 
Σinputs ⊙ weights to equal to zero, that can be achieved if either of the vectors/tensors are filled with 0, if all elements of the weight vector/tensor are 0 then that neuron recieves no input ever. But because we are using reLU the input vector/tensor can just be filled with zeros instead. Unfortunately for a network made of fully-connected layers that would mean no new information gets introduced to any neurons from that layer onwards, but we can get around this by assuming only some input elements = 0, and the weights that correspond to the non 0 inputs equal to zero (e.g. if input = [1,0,1,0] and weights = [0,1,0,1]) the sum of the Hadamard product of the 2 vectors/tensors is 0 even though not all elements are 0, because the weights that correspond to the non-zero inputs = 0. 
In other words we need sparse connections.

Going over the criteria, we need reLU, bias = 0, sparse connections and elements of input vector/tensor equal to 0, or more relaxed criteria of reLU/some variation of reLU, bias close to 0, sparse connections and some elements of input vector/tensor close to 0 for senarios where the information isn't needed indefinitely. In our opinion both are sets of reasonable enough criteria.

Alas we are still in the "spherical cows in a vacuum", phase. There are still questions that need to be answered. 

Firstly, will the network learn to keep the inputs to a neuron at 0 if it needs to that information some time later? We don't know, and I don't know how to or try to prove it with a "gedankenexperiment". (probably, but we can't prove it with any level of rigour) 

Secondly, will the network learn to keep bias close to 0 to preserve information? (probably, but we can't prove it with any level of rigour)

Thirdly, will these "memory neurons" even emerge through a training process? (absolutely no idea)

So we decided to try and design a new variant of RNN, that can fit the criteria and has a few other features. We are going to refer to these as "fastlane networks".

We ditched a layered structure, instead we have an order for the neurons to "fire" in and every neuron can have connections with any other neuron(no connections to itself, no point in doing so).
