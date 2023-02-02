import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# @desc: CountVectorizer instance
vec = CountVectorizer()


def get_top_n_words(corpus, n=None):
    """
    @desc: Get the top n words in the corpus.

    Args:
        corpus (list): List of strings.
        n (int): Number of top words to return.

    Returns:
        list: List of top n words.
    """
    # @desc: Get the main text from the corpus

    if corpus is not None and len(corpus) > 0:
        main_text = [text[0] for text in corpus]

        # @desc: Get the sentiment from the corpus
        sentiment = [text[1] for text in corpus]

        vec.set_params(**{"ngram_range": (1, 1)})

        # @desc: Get the vectorized text
        bag_of_words = vec.fit_transform(main_text)

        # @desc: Get the sum of the vectorized text
        sum_words = bag_of_words.sum(axis=0)

        # @desc: Get the words from the vectorized text
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]

        # @desc: Sort the words from the vectorized text
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # @desc: Get the top n words
        top_n_words = words_freq[:n]

        # @desc: Get the top n words and the sentiment of the words and frequency of the words
        top_n_words_sentiment = [
            {"id": i + 1, "word": word, "sentiment": sentiment[i],
             "frequency": f"{str(frequency)} times"}
            for i, (word, frequency) in enumerate(top_n_words)]

        return top_n_words_sentiment
    return []


# Data from dataset-all-combined-final.csv and split into train and test data 80/20
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset-all-combined-final.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"], shuffle=True)

# Get the top 10 words from the train data
top_n_words = get_top_n_words(train.values.tolist(), n=train.shape[0])

print(len(top_n_words))

# Get the top 10 words from the test data
top_n_words = get_top_n_words(test.values.tolist(), n=test.shape[0])

print(len(top_n_words))

# such as stemming and lemmatization to reduce the number of words in the corpus
# and to reduce the number of features in the model.

