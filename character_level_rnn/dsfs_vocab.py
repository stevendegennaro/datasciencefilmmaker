from typing import List
import json

class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int,str] = {}

        for word in (words or []):
            self.add(word)

    @property
    def size(self) -> int:
        return len(self.w2i)

    def add(self,word:str) -> None:
        if word not in self.w2i:
            word_id = len(self.w2i)
            self.w2i[word] = word_id
            self.i2w[word_id] = word
    
    def get_id(self, word:str) -> int:
        return self.w2i.get(word)

    def get_word(self, word_id: int) -> str:
        return self.i2w.get(word_id)

    def one_hot_encode(self, word: str) -> List:
        word_id = self.get_id(word)
        assert word_id is not None, f"unknown word {word}"
        return [1.0 if i == word_id else 0.0 for i in range(self.size)]

def save_vocab(vocab: Vocabulary, filename: str) -> None:
    print(f"Saving vocab to file: {filename}")
    with open(filename,"w") as f:
        json.dump(vocab.w2i,f)

def load_vocab(filename: str) -> Vocabulary:
    print(f"Loading vocab from file: {filename}")
    vocab = Vocabulary()
    with open(filename) as f:
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
        print("Vocab Loaded")
        return vocab