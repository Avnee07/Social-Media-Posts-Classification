import csv
from src.preprocess import cleaning
from src.bow import make_vocabulary
from src.bow import all_posts_to_vectors
from src.naiveBayes import NaiveBayes
import random

def load_data(path):
    posts = []
    labels = []
    
    with open(path, encoding="utf-8-sig") as file:
        dict = csv.DictReader(file)
        for row in dict:
            post = row['text']      
            label = row['Class']

            clean_words = cleaning(post)
            posts.append(clean_words)

            # posts.append(post)   
            labels.append(int(label))
    return posts, labels

# 2 lists -> posts and labels from the dataset 


posts, labels = load_data("Use this for data.csv")
# print(posts)
i = 1
# for post in posts:
#     print(i,post)
#     print('\n')
#     i = i + 1

vocab = make_vocabulary(posts)
print("Vocabulary size:", len(vocab))
print("Number of posts:", len(posts))

X = all_posts_to_vectors(posts, vocab)
y = labels

random.seed(42) 
combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)

spliting = int(0.8 * len(X))
X_train, y_train = X[:spliting], y[:spliting]
X_test, y_test = X[spliting:], y[spliting:]

model = NaiveBayes()
model.train(X_train, y_train)

correct = 0
for xt, yt in zip(X_test, y_test):
    pred = model.predict(xt)
    if pred == yt:
        correct += 1

accuracy = correct / len(X_test)
print("\nNaive Bayes Accuracy:", accuracy)
