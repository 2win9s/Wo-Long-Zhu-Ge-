SciFest@College AIT 2020 project by Kacper Gibas, Sam Kierans and Billy Yan.

apologies we do not know how to put in citations so we just stuck the links in where needed. sorry about that


Retaining information through arbitrary timesteps with 'simple' RNN?
Kacper Gibas, Sam Kierans and Billy Yan.



A basic RNN(https://www.researchgate.net/publication/243698906_Finite_State_Automata_and_Simple_Recurrent_Networks) achieves memory by receiving 2 sets of inputs;new information input and previous hidden state and returning 2 sets of outputs the output and the new hidden state.

LSTM(https://www.bioinf.jku.at/publications/older/2604.pdf) perhaps the most popular form of RNN uses hidden states and gates to retain information for arbitrary timesteps. This allows LSTMs ,when combined with other techniques such as transformers(https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) become state of the art in practically all areas of a.i. research where it is applicable(e.g. Natural Language processing: https://github.com/huggingface/transformers).    

But is a hidden state essential for retaining information for many timesteps?

Biological neural networks (brains) don't seem to use one; they probably don't send of its information to a separate entity in order to wipe or reset itself, I think we can all agree that the information stays in the brain and doesn't get wiped. In our brains certain regions within the neural network, through certain structures/mechanisms/properties combat decay and/or interference (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3971378/), allowing information to persist. 

Can we try something similiar with Artificial neural networks?
Perhaps, but there could be many more mechanisms/structures/properties of biological neural networks that haven't been discovered, or that we do not understand yet that allow it to retain information the way it does, but there is no harm in trying.
How will we retain information without a hidden state?
The obvious method is to just not wipe the neural network and just do all the calculations on top of neurons that aren't reset to 0, information from the previous timestep for free!!! 

But is it possible for that approach to retain information through arbitrary timesteps? 


















