class Vocabulary:
    def __init__(self, corpus) -> None:
        tokens = list(set(corpus.split(' ')))
        self.voc = {'<PAD>':0,'<BOS>':1,'<EOS>':2}
        
        #self.max_len = max_len
        for k in range(len(tokens)):
            self.voc[tokens[k]] = k + 3
        self.inv_voc = {v:k for k,v in self.voc.items()}

    def encode(self, sent: str):
        tokens = sent.split(' ')
        tokens = [self.voc[word] for word in tokens]
        tokens = [self.voc['<BOS>']] + tokens + [self.voc['<EOS>']]
        return tokens

    def decode(self, tokens) -> str:
        sent = ''
        for token in tokens:
            sent = sent + ' ' + self.inv_voc[token]
        return sent

