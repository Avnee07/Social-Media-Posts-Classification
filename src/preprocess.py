import re
import string
import spacy

nlp = spacy.load('en_core_web_sm')


def cleaning(df):
# lowercasing
    df = df.lower()

# df = re.sub(r"http\S+", "", text)  
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    df = url_pattern.sub('', df)

# removing punctations because after tokenization vocab will become larger for no reason. Also  Hello! and Hello will be treated differently
    exclude = string.punctuation
    df = df.translate(str.maketrans('','',exclude))

# not removing stopwords as it removes context in my case many times

    # words = text.split()
    words = nlp(df)
    # clean_words = [token.text for token in words if not token.is_space]
    clean_words = [token.lemma_ for token in words if not token.is_space]

    return clean_words