import csv
from src.preprocess import cleaning
from src.bow import make_vocabulary
from src.bow import all_posts_to_vectors
 
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

# print(X)
