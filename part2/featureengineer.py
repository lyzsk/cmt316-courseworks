import nltk
import operator
import numpy
from part2.processing import category_dic
from part2.processing import X_train, Y_train


stopwords = set(nltk.corpus.stopwords.words("english"))
extra_stopwords = [".", ",", ":", ";", "``", "''", "'", "%", "-", "$", "(", ")"]
for char in extra_stopwords:
    stopwords.add(char)

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# vocabulary = []

# function to get most n frequent words in the full dataset, which include all 5 categories, excluding stopwords
def get_frequent_words(article_list, n):
    frequency_dic = {}
    # print(len(article_list))
    # print(article_list.loc[0])
    # print(article_list.loc[1])
    for i in article_list.index:
        article = article_list.loc[i, "content"]
        # print(article)
        for sentence in nltk.tokenize.sent_tokenize(article):
            for token in nltk.tokenize.word_tokenize(sentence):
                word = lemmatizer.lemmatize(token).lower()
                if word in stopwords: continue
                if word in frequency_dic:
                    frequency_dic[word] += 1
                else:
                    frequency_dic[word] = 1
        sorted_list = sorted(frequency_dic.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_list)
        if len(sorted_list) <= n:
            print("no {} words in list, the number of words is: {}".format(n, len(sorted_list)))
            return sorted_list
        else:
            return sorted_list[:n]


# function to get vector according to vocabulary from articles
# feature 1 refers to get the number n most frequent words from all categories
def get_feature1_vector(vocabulary, article):
    vector = numpy.zeros(len(vocabulary))
    words = []
    for sentence in nltk.tokenize.sent_tokenize(article):
        for token in nltk.tokenize.word_tokenize(sentence):
            words.append(lemmatizer.lemmatize(token).lower())

    for i, word in enumerate(vocabulary):
        if word in words:
            vector[i] = words.count(word)
    return vector


# Function to get train data for feature 1
# feature 1 refers to get the number n most frequent words from all categories
def get_feature1_train_data(n):
    vocabulary = []

    words = get_frequent_words(X_train, n)
    for word, frequency in words:
        if word not in vocabulary:
            vocabulary.append(word)

    X_vector = []
    Y_vector = []
    for i in X_train.index:
        X_vector.append(get_feature1_vector(vocabulary, X_train.loc[i, "content"]))
        Y_vector.append(Y_train.loc[i])
    return X_vector, Y_vector, vocabulary


def get_feature2_vector(vocabulary, article):
    vector = numpy.asarray(len(vocabulary))
    words = []
    # get token from the first line of each article as title
    for token in nltk.tokenize.word_tokenize(nltk.tokenize.sent_tokenize(article)[0]):
        words.append(lemmatizer.lemmatize(token).lower())
    for i, word in enumerate(vocabulary):
        if word in words:
            vector[i] = words.count(word)
    return vector

def get_feature2_train_data(n):
    title_vocabulary = []
    for i in X_train.index:
        article = X_train.loc[i, "content"]
        # get token from the first line of each article as title
        for token in nltk.tokenize.word_tokenize(nltk.tokenize.sent_tokenize(article)[0]):
            word = lemmatizer.lemmatize(token).lower()
            if word in stopwords: continue
            if word not in title_vocabulary: title_vocabulary.append(word)

    X_vector = []
    Y_vector = []
    for i in X_train.index:
        X_vector.append(get_feature2_vector(title_vocabulary, X_train.loc[i, "content"]))
        Y_vector.append(Y_train.loc[i])
    return X_vector, Y_vector, title_vocabulary

