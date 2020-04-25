SciFest@College AIT 2020 project by Kacper Gibas, Sam Kierans and Billy Yan.



A 'simpler' RNN?

A basic RNN achieves memory by receiving 2 sets inputs the input and previous hidden state and returning 2 sets of outputs the output and the new hidden state.

Even LSTM used in most if not all of the state of the art RNNs employ a hidden state.
Why do we need to output a hidden state? 
One reason is because you can use it in attention& transformers, but those were ingenious ideas created around hidden states.
Before attention why did we have hidden states? 


Why do we need to store the information outside the neural network?


It seems like a counter intuitive way to retain information; you have to gather information from you current timestep, manipulate it into a hidden state/ add to a existing hiddenstate, 'reset' your neural network, feed the hiddenstate back in with new input.
Biological neural networks (brains) probably don't send information from the entire network to an external source, then wipe/reset the rest of the brain and then requests for the information for the next input and repeat.
To us it appears that we have hidden states because we 'wipe' or use a new blank neural network for every new timestep (we could be terribly wrong about this). By allowing information to stay in the neural network we might be able to see a/multiple 'memory regions' emerge that through certain structures/mechanisms combat decay and/or interference, which is what the brain seems to do (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3971378/). 


We then decided try and reason how a simple artificial neural network might be able to retain information through long series of timesteps, being able to retain information through many timesteps is key for most tasks; that is why LSTM is so widely used.
















