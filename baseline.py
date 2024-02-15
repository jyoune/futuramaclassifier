import sklearn.model_selection
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.metrics
import pandas
import nltk


def main():
    train_docs = pandas.read_csv('training_set.csv')
    dev_docs = pandas.read_csv('development_set.csv')
    test_docs = pandas.read_csv('test_set.csv')
    # strip each set of docs of punctuation and lowercase them. this is baseline for unigrams.
    train_docs["Line"] = train_docs["Line"].str.replace(r'[^\w\s]', '', regex=True).str.lower()
    dev_docs["Line"] = dev_docs["Line"].str.replace(r'[^\w\s]', '', regex=True).str.lower()
    test_docs["Line"] = test_docs["Line"].str.replace(r'[^\w\s]', '', regex=True).str.lower()
    # make sure each line column is a string. had errors for this before.
    train_docs["Line"] = train_docs["Line"].astype(str)
    dev_docs["Line"] = dev_docs["Line"].astype(str)
    test_docs["Line"] = test_docs["Line"].astype(str)
    # add a tokens column to each
    train_docs = add_tokenized_col(train_docs)
    dev_docs = add_tokenized_col(dev_docs)
    test_docs = add_tokenized_col(test_docs)
    # make dict lists to vectorize for each data set
    train_dict_list = featurize(train_docs, False, 1)
    dev_dict_list = featurize(dev_docs, False, 1)
    test_dict_list = featurize(test_docs, False, 1)
    # create vectorizer and vectorize each set
    vectorizer = sklearn.feature_extraction.DictVectorizer()
    feature_vec_train = vectorizer.fit_transform(train_dict_list)
    feature_vec_dev = vectorizer.transform(dev_dict_list)
    feature_vec_test = vectorizer.transform(test_dict_list)
    model = sklearn.naive_bayes.MultinomialNB(alpha=0, force_alpha=True)
    # create label arrays
    labels_train = train_docs["Character"].to_numpy()
    labels_dev = dev_docs["Character"].to_numpy()
    labels_test = test_docs["Character"].to_numpy()
    # train the baseline model with no smoothing and unigram features without counts
    model.fit(feature_vec_train, labels_train)
    # predict on dev set
    preds = model.predict(feature_vec_dev)
    print(sklearn.metrics.classification_report(labels_dev, preds))


def add_tokenized_col(data: pandas.DataFrame):
    data["Tokens"] = data["Line"].apply(lambda line: nltk.tokenize.word_tokenize(line))
    return data


def featurize(data, counts: bool, n: int):
    """
    this gets the list of feature dictionaries to use with DictVectorizer.
    for now, it works with unigrams for binary and count values. TODO: add bigrams etc.
    """
    feature_dict_list = []
    # so far this is for unigrams
    if not counts:
        for line in data["Tokens"]:
            line_dict = dict()
            for token in line:
                line_dict[token] = 1
            feature_dict_list.append(line_dict)
    else:
        for line in data["Tokens"]:
            line_dict = dict()
            for token in line:
                if token in line_dict:
                    line_dict[token] += 1
                else:
                    line_dict[token] = 1
            feature_dict_list.append(line_dict)
    return feature_dict_list


if __name__ == '__main__':
    main()
