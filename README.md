# notes 
Makemore is a character-level language model. Makemore models sequences of characters, thus it predicts the next character in a sequence. 

- Bigram models local structure in a dataset. It processes two characters at a time: based on character, predict the next character.

So, a bigram is a tuple of characters. Local structure can be modelled by counting the bigrams in the dataset: for every character, probabilities for the next character. The resulting names seem terrible, but performs way better than an 'untrained' (random, uniform distribution) model.

We trained the bigram language model by counting the number of ocurrences of any pairing and then normalizing (probability distribution). The elements are the parameters of the bigram language model, summmarizing the statistics of the bigrams. After training, we sample from the model: iteratively sample next character. 

How to evaluate quality of the model? (Average) negative log likelihood loss*. So, the goal (during training) is to find the parameters to minimize the loss.
*Maximize product of probabilites (likelihood) of the data wrt model parameters (probability that the model assigns to every bigram; if the model is perfect, then these should be all 1) = Maximize the sum of log probabilities (log likelihood, otherwise likelihood can get become tiny number) = minimizing negative log likelihood (loss function should be negative)

note: when a bigram doesn't appear in the dataset, then the loss is infinity (not desirable ofc). Apply model smoothing, adding fake counts.

- Cast bigram language model into neural network framework.

neural network will be bigram character-level language model: it receives single character as input, then neural network with some weights and gives back probability distribution over next character as output. Evaluate the output of the model using loss funtion. Train the network with gradient-based optimization to tune the weights of the network.

Prepare train set by having a character and label set. The label set contains the next characters. The characters are encoded as integers (index). How to feed in these examples into neural network? Input neuron shouldn't take on integer values, because they are multiplied with weights. Typical way to encode integers is one-hot encoding (as floats).

note: difference between torch.tensor and torch.Tensor. torch.tensor infers dtype; while torch.Tensor returns float.

At the moment, the neural network consists out of only a single layer of 27 neurons. With matrix multiplication it's very easy to evaluate the dot product between examples and neurons (neuron performs wx(+b)): F.e. perform matrix multiplication of 5 examples (5,27) and 27 neurons (27,27): this outputs the firing rate for every neuron on every example (5,27). 

The neural network outputs logits that can be interpreted as log counts. Then perform softmax. So, these are exponentiated to get the counts and then normalized to get the probabilities. This is the forward pass.

How can we tune the weights to achieve better outputs? Evaluate output (probability that model assigned to label character) by computing negative log likelihood for every example and then averaging it. So, minimize the loss by tuning the weights by computing the gradients of the loss wrt the weights. This is very comparable in micrograd, but using pytorch.

The loss of the neural network is about the same as the loss of the model before, because we don't take in any additional information. The explicit approach (counting the pairs) for a bigram language model captures the information very well without needing to perform gradient-based optimization. However, the gradient-based approach is very flexible. In a next step, the neural network can be made more complex: F.e. by taking mulltiple previous characters, using transformers...

https://pytorch.org/docs/stable/notes/broadcasting.html keepdim

- Multi-layer perceptron model 
Problem with previous approach: context of only a single character. If more characters are added as contexts, the matrix becomes exponentially bigger. So, try MLP: [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

Difference with bigram character-level language model: it receives multiple characters as input instead of a single character. 

The dataset is rebuild by using a block size (the amount of characters used as context). So, the character set is expanded to a number of characters that is equal to the block size. In the next step, the embedding lookup table is created to represent the characters (which are represented as integers) in a lower-dimensional space. 

So, the input to the hidden layer is a number of characters (equal to the block size) that are represented as low-dimensional vectors instead of integers or one-hot encoding. The hidden layer thus has following dimenion: number of neurons (this is a hyperparameter*) x (block size x vector dimension). The input characters are concatenated to fit the hidden layer.

*hyperparameters are parameters that are up to the desginer of the network (f.e. the size of the hidden layer)

The output layer has a dimension: number of neurons x the number of characters. Again, the output are logits. So, these need to be exponentiated to create fake counts and finally normalized to get a probability distribution over the characters. The model is trained with backpropagation and the negative log likelihood like the bigram model.

The last steps (from logits to loss) can be done much more efficiently using cross-entropy. Why? 1. the forward pass can be made much more efficient (no intermediate tensors); 2. the backward pass can be made much more efficient; 3. more numerically well-behaved (substract the max positive value from the logits)

The network is trained on a small dataset, the loss will decrease a lot. This indicates that the network is overfitting the data. The network can't completely overfit the data (0 loss), because some examples have different labels in the dataset. In the case where a unique input has a unique output, the network performs very well.

when training on the full dataset, the forward and backward pass take a lot of time. This can be solved by using mini-batches: randomly select portions of the data (mini-batch) and perform the forward and backward pass only on these mini-batches. When working with mini-batches, the quality of the gradient is lower (the direction is less reliable). But it's much better to have an approximate gradient and make more steps, then to calculate the exact gradient with fewer steps. 

The learning rate defines the size of the steps the network takes during training. A very low learning rate (small steps) means the loss decreases slowly. A very high learning rate means the loss can explode (unstable). The ideal learning rate is somewhere in between these two figures. Train the network with different learning rates in between and check which gets the best resulting loss. Learning rate decay (by factor of 10 for example).



# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT
