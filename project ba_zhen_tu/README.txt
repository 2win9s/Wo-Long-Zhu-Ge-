SciFest@College AIT 2020 project by Kacper Gibas, Sam Kierans and Billy Yan.



A 'simpler' RNN?

A basic RNN achieves memory by receiving 2 sets inputs the input and previous hidden state and returning 2 sets of outputs the output and the new hidden state.

Even LSTM used in most if not all of the state of the art RNNs employ a hidden state. When we first saw this concept of hidden states we asked the question:
Why do we need to output a hidden state? 
One reason is because you can use it in attention& transformers, but those were ingenious ideas created around hidden states.
We then started asking more and more questions.
Before attention why did we have hidden states? 


Why do we need to store the information outside the neural network?


Do biological neural networks use the same technique?


To us it seemed like a counter intuitive way to retain information; you have to gather information from you current timestep, manipulate it into a hidden state/ add to a existing hiddenstate, 'reset' your neural network, feed the hiddenstate back in with new input.
Biological neural networks (brains) probably don't send information from the entire network to an external source, then wipe/reset the brain and then get the information back and uses it as part of input to the neurons.
We just couldn't understand why artificial neural networks have to rely on hidden states, when biological neural networks probably don't.

From there we wondered if we allow information to stay in the neural network perhaps we will see a/multiple 'memory regions' emerge that through certain structures/mechanisms combat decay and/or interference, which is what the brain seems to do (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3971378/). 


We then decided try and reason how a simple artificial neural network might be able to retain information through long series of timesteps, being able to retain information through many timesteps is key for many difficult tasks; the reason for LSTM's popularity.
















