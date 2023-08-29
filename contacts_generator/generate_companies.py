import json
from pprint import pprint
from dsfs_vocab import Vocabulary,save_vocab
from dsfs_deep import (SimpleRnn,Linear,Momentum,Model,SoftMaxCrossEntropy,
						softmax,sample_from,GradientDescent,save_weights,load_weights)
import random
import tqdm
import numpy as np

np.random.seed(1)
random.seed(1)

def get_company_names():
	exchanges = ['nyse','nasdaq']
	names = []
	for exchange in exchanges:
		filename = 'data/' + exchange +'.json'
		with open(filename, "r", encoding='utf-8') as f:
			data = json.load(f)
		names.extend([company['name'] for company in data])
		names = sorted(list(set(names)))
		##### Clean the names to remove ones that have weird punctuation ######
	return names

names = get_company_names()

# Define the start and end characters of every name
START = "^"
STOP = "#"

def train(batchsize: int, n_epochs: int):
	for epoch in range(n_epochs):
		random.shuffle(names)
		batch = names[:batchsize]
		epoch_loss = 0
		for name in tqdm.tqdm(batch):
			rnn1.reset_hidden_state()
			rnn2.reset_hidden_state()
			name = START + name + STOP
			for prev,nexts in zip(name,name[1:]):
				inputs = vocab.one_hot_encode(prev)
				target = vocab.one_hot_encode(nexts)
				predicted = model.forward(inputs)
				epoch_loss += model.loss.loss(predicted,target)
				gradient = model.loss.gradient(predicted,target)
				model.backward(gradient)
				model.optimizer.step(model)
		print(epoch,epoch_loss,generate(vocab))
	save_weights(model,weightfile)

def generate(vocab: Vocabulary, seed_char: str = START, max_len: int = 160) -> str:
	rnn1.reset_hidden_state()
	rnn2.reset_hidden_state()
	output = [seed_char]

	while output[-1] != STOP and len(output) < max_len:
		this_input = vocab.one_hot_encode(output[-1])
		predicted = model.forward(this_input)
		probabilities = softmax(predicted)
		next_char_id = sample_from(probabilities)
		output.append(vocab.get_word(next_char_id))

	return ''.join(output[1:-1])

tr = True    # train first
cont = True    # continue training from previous weights
gen = True      # generate names when done training

# create the vocabulary object
vocab = Vocabulary([c for name in names for c in name])
vocab.add(START)
vocab.add(STOP)

weightfile = "weights/weights.txt"
vocabfile = "weights/vocab.txt"

save_vocab(vocab, vocabfile)

# Set up neural network
HIDDEN_DIM = 32
rnn1 = SimpleRnn(input_dim=vocab.size,hidden_dim=HIDDEN_DIM)
rnn2 = SimpleRnn(input_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM)
linear = Linear(input_dim=HIDDEN_DIM,output_dim=vocab.size)
loss = SoftMaxCrossEntropy()
optimizer = Momentum(learning_rate = 0.01,momentum=.9)
model = Model([rnn1,rnn2,linear],loss,optimizer,[],[])

# If this is a continuation of a previous training session,
# load the weights from that session
if cont: load_weights(model,weightfile)

# Train the network
batchsize = 100   # number of names to use for each round of training
batchsize = len(names)
n_epochs = 30      # number of rounds of training
if tr: train(batchsize, n_epochs)

# Generate new names
if gen:
	generated_names = []
	for _ in range(100):
		newname = generate(vocab)
		# Make sure this isn't just one of the names in the training data
		if newname not in names: generated_names.append(newname)
	print(generated_names)
	print(len(generated_names))

with open('generated_test.txt',"a") as f:
	json.dump(generated_names,f)


