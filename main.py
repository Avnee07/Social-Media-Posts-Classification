import csv
from src.preprocess import cleaning
 
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
for post in posts:
    print(post)
    print('\n')

#print("Total Posts:", len(posts))
