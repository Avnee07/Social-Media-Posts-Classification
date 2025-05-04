
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab_size = 0

    def train(self, X, y):
       
        label_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_docs = len(y)
        self.vocab_size = len(X[0])  

        for vector, label in zip(X, y):
            label_counts[label] += 1
            for i, count in enumerate(vector):
                word_counts[label][i] += count

        self.class_prob = {
            label: count / total_docs
            for label, count in label_counts.items()
        }

        # Laplace smoothing
        self.word_probs = {}
        for label in label_counts:
            total_words = sum(word_counts[label].values())
            self.word_probs[label] = {}
            for i in range(self.vocab_size):
                count = word_counts[label][i]
                self.word_probs[label][i] = (count + 1) / (total_words + self.vocab_size)

    def predict(self, vector):
       
        # print("Vector:", vector)
       
        log_probs = {}
        for label in self.class_prob:
            log_prob = math.log(self.class_prob[label])
            for i, count in enumerate(vector):
                if count > 0:
                    log_prob += count * math.log(self.word_probs[label].get(i, 1 / self.vocab_size))
            log_probs[label] = log_prob
        return max(log_probs, key=log_probs.get)
