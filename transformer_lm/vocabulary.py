from lib2to3.pgen2 import token
import numpy as np

class Vocabulary:
    def __init__(self, corpus) -> None:
        tokens = list(set(corpus.split(' ')))
        self.voc = {'<PAD>':0,'<SEP>':1,'<MASK>':2,'<CLS>':3}
        
        for k in range(len(tokens)):
            self.voc[tokens[k]] = k + 4
        self.inv_voc = {v:k for k,v in self.voc.items()}

    def encode(self, sent: str):
        tokens = sent.split(' ')
        tokens = [self.voc[word] if word in self.voc else 0 for word in tokens]
        tokens = [self.voc['<CLS>']] + tokens + [self.voc['<SEP>']]
        return tokens

    def encode_masked(self, sent: str):
        tokens = sent.split(' ')
        tokens = np.array([self.voc[word] if word in self.voc else 0 for word in tokens])
        mask = np.random.rand(tokens.shape[0])
        tokens = (mask > 0.15) * tokens
        tokens[tokens == 0] = self.voc['<MASK>']
        tokens = [self.voc['<CLS>']] + list(tokens) + [self.voc['<SEP>']]
        return tokens

    def decode(self, tokens) -> str:
        sent = ''
        for token in tokens:
            sent = sent + ' ' + self.inv_voc[token]
        return sent

class VocabularyCharLevel:
    def __init__(self, corpus) -> None:
        tokens = set(list(corpus))
        self.voc = {'<PAD>':0,'<BOS>':1,'<EOS>':2}
        
        for k in range(len(tokens)):
            self.voc[tokens[k]] = k + 3
        self.inv_voc = {v:k for k,v in self.voc.items()}

    def encode(self, sent: str):
        tokens = list(sent)
        tokens = [self.voc[word] for word in tokens]
        tokens = [self.voc['<BOS>']] + tokens + [self.voc['<EOS>']]
        return tokens

    def decode(self, tokens) -> str:
        sent = ''
        for token in tokens:
            sent = sent + self.inv_voc[token]
        return sent
