import csv
 
def load_data(path):
    posts = []
    labels = []
    
    with open(path, encoding="utf-8-sig") as file:
        dict = csv.DictReader(file)
        for row in dict:
            post = row['text']      
            label = row['Class']

            posts.append(post)   
            labels.append(int(label))
    return posts, labels

# 2 lists -> posts and labels from the dataset 


posts, labels = load_data("Use this for data.csv")
#print(posts)
#print("Label:", labels[0])
#print("Total Posts:", len(posts))
