SciFest@College AIT 2020 project by Kacper Gibas, Sam Kierans and Billy Yan.



A 'simpler' RNN?

A basic RNN achieves memory by receiving 2 sets inputs the input and previous hidden state and returning 2 sets of outputs the output and the new hidden state.

Even LSTM used in most if not all of the state of the art RNNs employ a hidden state.
Why do we need to output a hidden state?
One reason is because you can use it in attention& transformers, but those were ingenious ideas created around hidden states.
Before attention why did we have hidden states?

It seems like a counter intuitive way to retain information; you have to gather information from you current timestep, manipulate it into a hidden state/ add to a existing hiddenstate, 'reset' your neural network, feed the hiddenstate back in with new input.
Biological neural networks (brains) probably don't work by sending information to one particular region then resetting the rest of the brain and then sending the information back when new input arrives. What they probably do is retain information from the current time by not resetting itself. By not wiping the slate clean everytime there is a new input you quite naturally, have the information from the last timestep at the network's disposal without any extra work.
