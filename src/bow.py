def make_vocabulary(posts):
    unique_words = set()
    
    
    for post in posts:
        for word in post:
            unique_words.add(word)
    
    word_to_index = {}
    for i, word in enumerate(sorted(unique_words)):
        word_to_index[word] = i
    
    return word_to_index


def post_to_vector(post_words, vocab):
    vector = [0] * len(vocab)

    for word in post_words:
        if word in vocab:
            index = vocab[word]
            vector[index] = vector[index] + 1  
    
    return vector

def all_posts_to_vectors(posts, vocab):
    vector_list = []
    
    for post in posts:
        vector = post_to_vector(post, vocab)
        vector_list.append(vector)
    
    return vector_list