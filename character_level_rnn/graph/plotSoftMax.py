import json
from dsfs_vocab import load_vocab
from matplotlib import pyplot as plt

with open('softmax.json', 'r') as f:
    softmax = json.load(f)

vocab = load_vocab('weights/lastname_vocab.txt')
print(softmax)
print(vocab.w2i.keys())
print(len(softmax), len(vocab.w2i.keys()))
data = {}
for i in range(len(vocab.w2i.keys())):
    data[list(vocab.w2i.keys())[i]] = softmax[i]
print(data)

plt.bar(list(vocab.w2i.keys()),softmax)
plt.show()