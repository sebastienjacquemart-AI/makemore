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

The dataset is rebuild by using a block size (the amount of characters used as context). So, the character set is expanded to a number of characters that is equal to the block size. In the next step, the embedding lookup table is created to represent the characters (which are represented as integers) in a lower-dimensional space. These lookup tables are also parameters that will be trained. 

So, the input to the hidden layer is a number of characters (equal to the block size) that are represented as low-dimensional vectors instead of integers or one-hot encoding. The hidden layer thus has following dimenion: number of neurons (this is a hyperparameter*) x (block size x vector dimension). The input characters are concatenated to fit the hidden layer.

*hyperparameters are parameters that are up to the desginer of the network (f.e. the size of the hidden layer)

The output layer has a dimension: number of neurons x the number of characters. Again, the output are logits. So, these need to be exponentiated to create fake counts and finally normalized to get a probability distribution over the characters. The model is trained with backpropagation and the negative log likelihood like the bigram model.

The last steps (from logits to loss) can be done much more efficiently using cross-entropy. Why? 1. the forward pass can be made much more efficient (no intermediate tensors); 2. the backward pass can be made much more efficient; 3. more numerically well-behaved (substract the max positive value from the logits)

The network is trained on a small dataset, the loss will decrease a lot. This indicates that the network is overfitting the data. The network can't completely overfit the data (0 loss), because some examples have different labels in the dataset. In the case where a unique input has a unique output, the network performs very well.

when training on the full dataset, the forward and backward pass take a lot of time. This can be solved by using mini-batches: randomly select portions of the data (mini-batch) and perform the forward and backward pass only on these mini-batches. When working with mini-batches, the quality of the gradient is lower (the direction is less reliable). But it's much better to have an approximate gradient and make more steps, then to calculate the exact gradient with fewer steps (Even if each mini-batch update is a little "wrong," it usually still moves you roughly toward the minimum. Over many small steps, the errors average out, and you make faster progress per unit of computation.). 

The learning rate defines the size of the steps the network takes during training. A very low learning rate (small steps) means the loss decreases slowly. A very high learning rate means the loss can explode (unstable). The ideal learning rate is somewhere in between these two figures. Train the network with different learning rates in between and check which gets the best resulting loss. Learning rate decay (by factor of 10 for example).

If the capacity of the neural network grows (more parameters), it becomes more capable of overfitting the dataset: the loss on the training set is very low (approx 0), but the loss on data points outside of the training set will be much higher. It memorizes the data from the training set. Split dataset in three splits: train, dev/validation and test split (80-10-10). The training set is used to optimize the parameters of the model; the development/validation split is used for development of hyperparameters of the model; test split is used to evaluate the performance of the network. When training and development loss are close to each other, it indicates that the network is not overfitting, but underfitting. So, the model is too small and performance improvements can be achieved by scaling up the model. When they are not close to each other, it indicates that the network is overfitting. 

If the batch size is set too low, then the gradient might become noisy. So, increase the batch size to be able to optimize more properly. 

- Before we move on to other models, we have to understand the activations and gradients of the multi-layer perceptron model during training. Let's look at a few problems with initialization first:

1. Initialization of logits. Rough idea of the initial loss using the definition of the loss and the problem at hand: The initial output probablity distribution should be uniform (random). So, the probability for any character should be 1/27. The negative log likelihood of this number is the expected initial loss. In case of high initial loss, the network is very confidently wrong (extreme logits). The output logits should be close to zero. This can be achieved by 1. zero bias; 2. scale down weights of output layer. 

2. Initialization of activation of hidden states. Tanh-function squashes number between a range of 1 and -1. When the distribution of the tanh-values (after activation) mostly take on values of 1 or -1, then there's a problem. The cause can be found in the distribution of the values before activation which is very broad. Why is this a problem? If many of the neurons are in the flat regions of 1 or -1, then the backward gradients will get destroyed at this layer*. A dead neuron happens when all the examples land in the tale region (no example activates the neuron in the active part of the neuron), this way the neuron will never learn. This problem is true for many activation functions, except Leaky ReLu. This problem can be fixed by scaling down the biases and the weights. 

*The more the output is in the flat tales, the more the gradient is squashed. If output of the activation function for a certain neuron is in the flat region of the tanh-function, then the weights and the biases of this neuron will stop influencing the loss and the gradient will be destroyed. So, no matter how the weights or biases are changed, it doesn't activate the neuron in the active part of the tanh or it doesn't infleunce the output and the loss. 

In the case of a shallow network (1/2 layers), the network can be forgiving for these problems. However, when the network gets deeper, these problems stack up. 

There's a principles way to set these scales. The activations shouldn't grow to infinity or shrink to zero. They should stay unit gaussian (standard deviation of about 1) throughout the entire neural network. If the weights are scaled up, then the gaussian grows in standard deviation (the numbers in the output take on more extreme values). If the weights are scaled down, the standard deviation shrinks. So, what should the weights be multiplied with to get a standard deviation of 1? Divide the weights by the square root of the fan-in. The fan-in is the number of inputs to the node. Initialize the weights using kaiming normal, which mostly depends on the non-linearity. It has become less important to initialize the networks right due to some modern innovations such as residual connecitons, normalization layers (batch, layer....), optimizers (Adam,...). In practice, normalize weights by skaiming normal. So, gain divided by square root of fan-in, where the gain depends on the non-linearity. If the activations are not gaussian, you can run into a lot of problems with the non-linearities. 

The pre-activations (hidden states, outputs of node) shouldn't be too large (then non-linearity is saturated) or too small (then non-linearity does nothing). So, easy solution by making the hidden states gaussian: substract the mean and divide by the standard deviation of the neurons. The problem is that the hidden states should be gaussian at initialization, but they shouldn't be forced to be gaussian always, because the backpropagation should be able to move the distribution around. So, scale the normalized inputs by some gain (initialized as ones) and offsetting them by some bias (initialized as zeros). So, each neurons firing value is unit gaussian at initialization, during optimization backpropagation can change the gain and bias. 

In a simple neural network, there's not much of performance improvement (compared to calculating the scale using kaimin normal). In a deeper neural network, it's way more difficult to tune the scales of the weights matrices to make the activations roughly gaussian. It will be much easier using batch normalization. However, the stability of batch normalization comes at a cost. Before, deterministic process: activations and logits are calculated for a single example (or in batches). Now, the logits are a function of all the examples in the batch. This turns out to be good as it makes it harder for the neural network to overfit these concrete examples. But in general, people don't like examples to be coupled. How to do the inference? Calculate mean and standard deviation once on the entire training set or keep  a running mean and running deviation, these are not part of backpropagation, but are updated on the side. These can then be used at prediction to be able to forward individual examples. Batch normalization has its own bias. So, no need to add bias. 

A real-life example: resnet50. This is a convolutional neural network that excels at classifying images. This network is built out of convolutional layers. These are similar to linear layers, but used for images: the linear multiplication are performed on overlapping patches. There's a repeating pattern in the network: conv, batch norm, relu. 

Let's look at gradient and activation statistics in case of no batch normalization and the relation to the gain. In case of no gain, the standard deviation of the first layer is 1, but due to the non-linearity layers, the distribution of the layers gets squashed and the standard deviation shrinks (low saturation, where saturation is the amount of values of the neurons that are in the tales of the non-linearity). In case of right gain, the standard deviation of the first layer is high, but it will stabilize in the next layers (decent saturation). In case of low gain, the standard deviation of the gradients will grow. In case of high gain, the standard deviation of the gradients shrinks. In case of right gain, the distribution of the gradients stays roughly the same. So, why do we even need non-linearity layers? No matter how many linear layers are stacked, the result will be a linear function. Also important to look at the statistics of the parameters. Important new statistics: 1. gradient-to-parameter ratio, which shows the scale of the gradient compared to the scale of the actual weights. Ideally, this ratio is small; 2. update-to-data ratio over iterations, which shows the standard deviation of the update (learning rate*gradient) compared to the log of the standard deviation of the weights (how great are the updates to the weights?) over iterations. Ideally, these ratios are around 1e-3. If this ratio is higher, then the weights are undergoing a lot of change. In this case, the last layer has a high ratio. Ideally, all these statistics are roughly symmetrical for all the layers. These plots are a good way to bring miscalibrations to your attention. 

With batch normalization the network is way more robust to the gain of the linear layers. It's not a free pass, the update-to-data ratio can be messed up...

- Let's implement backpropagation manually for the mlp model. We did this in micrograd, but on element-level. Now, we implement it on tensor-level.. (SHOULD RE-WATCH THIS VIDEO)

The output of the network are log probabilities for all possible characters for every example in the batch (logprobs). The loss is the negative mean of the logprobs for the correct (groundtruth) characters for every example in the batch. Calculate the gradient as the derivative of the loss with respect to all the elements in logprobs (dlogprobs). dlogporbs will be -1/n for all correct elements. dlogprobs will be zero for all incorrect incorrect elements, because they don't feed into the loss (if these numbers are changed, then the loss doesn't change). 

The output of the network are logits. The cross-entropy loss is claculated using these logits: For every example, 1. the logits are normalized by substracting the max logit (no influence on probabilities, this is done for stabilit); 2. the normalized logits are exponentiated to get the counts; 3. the counts are divided by the sum of the counts to get the probabilities; 4. the log of the probabilities are taken; 5. the negative mean of the logprobs for the correct examples results in the loss. 

During this exercise, calculate gradient step-by-step: mainly chain rule as in micrograd. However, be careful as some operations are two operations combined: For example, when the counts are normalized by dividing by the sum of the count

- 


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
