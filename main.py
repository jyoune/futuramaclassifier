import sklearn.model_selection
import sklearn.feature_extraction
import sklearn.naive_bayes
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import pandas
import nltk
from collections import Counter, defaultdict
from nltk import ngrams


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
    # add appropriate n-gram columns to each
    train_docs = add_ngram_feature_cols(train_docs)
    dev_docs = add_ngram_feature_cols(dev_docs)
    test_docs = add_ngram_feature_cols(test_docs)
    # get the top 100 bi and trigrams for each set
    for docs in [train_docs, dev_docs, test_docs]:
        for i in range(2, 4):
            docs = find_top_100(docs, i)
    print(train_docs.loc[0, :])
    # run_baseline(train_docs, dev_docs, test_docs)
    # run_configs(train_docs, dev_docs, test_docs)
    # based on results in output.txt i take the 4 best performing models and do the test output.
    test_best_models(train_docs, dev_docs, test_docs)


def run_baseline(train_docs, dev_docs, test_docs):
    # make dict lists to vectorize for each data set
    train_dict_list = featurize_single_type(train_docs, False, 1)
    dev_dict_list = featurize_single_type(dev_docs, False, 1)
    test_dict_list = featurize_single_type(test_docs, False, 1)
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
    with open("output.txt", "a") as file:
        print("Baseline- unigrams MultinomialNB no smoothing", file=file)
        print("accuracy: ", sklearn.metrics.accuracy_score(labels_dev, preds), file=file)
        print(sklearn.metrics.classification_report(labels_dev, preds), file=file)


#this does the prediction on dev set.
def run_configs(train_docs, dev_docs, test_docs):
    # create label arrays
    labels_train = train_docs["Character"].to_numpy()
    labels_dev = dev_docs["Character"].to_numpy()
    labels_test = test_docs["Character"].to_numpy()
    # predict on dev set
    # TODO: actually write the grid search. may be worth to offload to another method.
    # TODO: Note that with zero smoothing, bigrams and trigrams are quite bad. Remember to figure out
    # TODO: another solution to storing the values and finding a maximum. Nested for loops where loop
    # TODO: variable is part of a key maybe? Or nested dictionary? Not sure.
    # first do NB, and set a range for alpha.
    # alpha_range = [x / 100 for x in range(10, 110, 10)]
    alpha_range = [x / 100 for x in range(40, 105, 5)]
    best_nb_config = grid_searching(train_docs,dev_docs, test_docs, labels_train, labels_dev, alpha_range)
    # now do LR and set a range for big_c.
    '''
    c_range = [x / 10 for x in range(5, 55, 5)]
    best_lr_config = grid_searching(train_docs, dev_docs, test_docs, labels_train, labels_dev,
                                    c_range, True)
                                    
    c_range = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 5, 10]
    best_lr_config = logistic_tighter_grid_search(train_docs, dev_docs, test_docs, labels_train, labels_dev, c_range)
    '''
    c_range = [0.15, 0.25, 0.35, 0.45, 0.55]
    best_lr_config = logistic_tighter_grid_search(train_docs, dev_docs, test_docs, labels_train, labels_dev, c_range)
    # From here, four best models are selected and tested in test_best_models.


# this method tests predictions of the four best models on the test set.
def test_best_models(train_docs, test_docs, dev_docs):
    """
    The best four models are as follows:
    MNB:
    Unigram +  use_bigrams: True use_trigrams: False Counts: False use100: True alpha: 0.7
    Unigram +  use_bigrams: True use_trigrams: False Counts: True use100: True alpha: 0.75

    LR:
    ngram: 1 counts: False big_c: 0.5
    Unigram +  use_bigrams: True use_trigrams: False Counts: False use100: False big_c: 0.45
    """
    # create label arrays
    labels_train = train_docs["Character"].to_numpy()
    labels_dev = dev_docs["Character"].to_numpy()
    labels_test = test_docs["Character"].to_numpy()
    config_strings = ["MNBUnigram +  use_bigrams: True use_trigrams: False Counts: False use100: True alpha: 0.7",
                      "MNBUnigram +  use_bigrams: True use_trigrams: False Counts: True use100: True alpha: 0.75",
                      "LRngram: 1 counts: False big_c: 0.55",
                      "LRUnigram +  use_bigrams: True use_trigrams: False Counts: False use100: False big_c: 0.45"]
    config_accuracy = {}
    # first model
    preds = model_train_predict(0.7, train_docs, dev_docs,test_docs, labels_train, False,True,
                                False, True, use_test=True)
    config_accuracy[config_strings[0]] = sklearn.metrics.accuracy_score(labels_test, preds)
    # second model
    preds = model_train_predict(0.75, train_docs, dev_docs, test_docs, labels_train, True, True,
                                False, True, use_test=True)
    config_accuracy[config_strings[1]] = sklearn.metrics.accuracy_score(labels_test, preds)
    # third model, now LR and unigram only
    preds = logistic_train_predict(0.55, train_docs, dev_docs, test_docs, labels_train, False,
                                   False, False, False, use_single_n=1, use_test=True)
    config_accuracy[config_strings[2]] = sklearn.metrics.accuracy_score(labels_test, preds)
    # fourth model
    preds = logistic_train_predict(0.45, train_docs, dev_docs, test_docs, labels_train, False,
                                   True,False, False, use_test=True)
    config_accuracy[config_strings[3]] = sklearn.metrics.accuracy_score(labels_test, preds)
    # print results
    with open('output.txt','a') as file:
        max_acc = 0
        max_key = ""
        for key in config_accuracy:
            model_string = key + " " + str(config_accuracy[key])
            print(model_string, file=file)
            if config_accuracy[key] > max_acc:
                max_acc = config_accuracy[key]
                max_key = key
        print("Best: ", file=file)
        print(max_key + " " + str(config_accuracy[max_key]), file=file)





# returns a string including best feature config and accuracy. also prints results to output.txt
def grid_searching(train_docs, dev_docs, test_docs, labels_train, labels_dev, hyper_range, use_reg_model: bool = False):
    # note that there are two sets of alpha here, the first marks the initial sweep.
    # second marks the more specific one.
    # single feature grid search
    accuracy_dict = defaultdict(dict)
    best_hyper_dict = {}
    for use_single_n in [1, 2, 3]:
        for use_counts in [True, False]:
            this_big_key = "ngram: " + str(use_single_n) + " counts: " + str(use_counts)
            for hyper in hyper_range:
                if not use_reg_model:
                    preds = model_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train, use_counts,
                                                False,False, False, use_single_n)
                else:
                    preds = logistic_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train, use_counts,
                                                   False, False, False, use_single_n)
                this_accuracy = sklearn.metrics.accuracy_score(labels_dev, preds)
                accuracy_dict[this_big_key][hyper] = this_accuracy
            best_hyper_dict[this_big_key] = max(accuracy_dict[this_big_key], key=accuracy_dict[this_big_key].get)
    for key in best_hyper_dict:
        print(key + " " + str(best_hyper_dict[key]) + " " + str(accuracy_dict[key][best_hyper_dict[key]]))
    # multi feature sets
    for use_counts in [True, False]:
        for use_bigrams in [True, False]:
            for use_trigrams in [True, False]:
                for use_100 in [True, False]:
                    this_big_key = ("Unigram + " + " use_bigrams: " + str(use_bigrams) + " use_trigrams: " +
                                    str(use_trigrams) + " Counts: " + str(use_counts) + " use100: " + str(use_100))
                    for hyper in hyper_range:
                        if not use_reg_model:
                            preds = model_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train,
                                                        use_counts, use_bigrams, use_trigrams, use_100)
                        else:
                            preds = logistic_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train,
                                                           use_counts, use_bigrams, use_trigrams, use_100)
                        this_accuracy = sklearn.metrics.accuracy_score(labels_dev, preds)
                        accuracy_dict[this_big_key][hyper] = this_accuracy
                    best_hyper_dict[this_big_key] = max(accuracy_dict[this_big_key],
                                                        key=accuracy_dict[this_big_key].get)
    return print_results(accuracy_dict, best_hyper_dict, use_reg_model)


# based on previous grid search i've found that logistic is worse with counts. also not affected much
# by use_100 when both bi and trigrams are present. these are also the most intensive (highest # features
# without using top 100) so i've written a new gridsearch for these parameters and new values of C.
def logistic_tighter_grid_search(train_docs, dev_docs, test_docs, labels_train, labels_dev, hyper_range):
    use_reg_model = True
    accuracy_dict = defaultdict(dict)
    best_hyper_dict = {}
    for use_single_n in [1, 2, 3]:
        for use_counts in [True, False]:
            this_big_key = "ngram: " + str(use_single_n) + " counts: " + str(use_counts)
            for hyper in hyper_range:
                preds = logistic_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train, use_counts,
                                               False, False, False, use_single_n)
                this_accuracy = sklearn.metrics.accuracy_score(labels_dev, preds)
                accuracy_dict[this_big_key][hyper] = this_accuracy
            best_hyper_dict[this_big_key] = max(accuracy_dict[this_big_key], key=accuracy_dict[this_big_key].get)
    for key in best_hyper_dict:
        print(key + " " + str(best_hyper_dict[key]) + " " + str(accuracy_dict[key][best_hyper_dict[key]]))
    # multi feature sets
    for use_counts in [False]:
        for use_bigrams in [True, False]:
            for use_trigrams in [True, False]:
                for use_100 in [True, False]:
                    if use_trigrams:
                        use_trigrams = use_100
                    this_big_key = ("Unigram + " + " use_bigrams: " + str(use_bigrams) + " use_trigrams: " +
                                    str(use_trigrams) + " Counts: " + str(use_counts) + " use100: " + str(use_100))
                    for hyper in hyper_range:
                        preds = logistic_train_predict(hyper, train_docs, dev_docs, test_docs, labels_train,
                                                       use_counts, use_bigrams, use_trigrams, use_100)
                        this_accuracy = sklearn.metrics.accuracy_score(labels_dev, preds)
                        accuracy_dict[this_big_key][hyper] = this_accuracy
                    best_hyper_dict[this_big_key] = max(accuracy_dict[this_big_key],
                                                        key=accuracy_dict[this_big_key].get)
    return print_results(accuracy_dict, best_hyper_dict, use_reg_model)


# prints but also returns because it's easier here
def print_results(accuracy_dict, best_hyper_dict, use_big_c: bool = False):
    best_feature_accs = {}
    if use_big_c:
        hyper = " big_c: "
    else:
        hyper = " alpha: "
    with open("output.txt", "a") as file:
        for key in best_hyper_dict:
            print(key + " " + hyper + str(best_hyper_dict[key]) + " " + str(accuracy_dict[key][best_hyper_dict[key]]), file=file)
            best_feature_accs[key] = accuracy_dict[key][best_hyper_dict[key]]
        print("Best: ", file=file)
        max_key = max(best_feature_accs, key=best_feature_accs.get)
        best_config_with_acc = str(max_key) + hyper + str(best_hyper_dict[max_key]) + " " + str(best_feature_accs[max_key])
        print(best_config_with_acc, file=file)
        return best_config_with_acc


# returns model predictions. note four parameters are bools and the last one is an int that defaults to 0
# also note this is for MultinomialNB only
# TODO: refactor so this is neater.
def model_train_predict(alpha, train_docs, dev_docs, test_docs, labels_train, counts, use_bigrams, use_trigrams, top_100, use_single_n: int = 0, use_test:bool = False):
    # bigrams are used by default in featurize_multi_type, and since they're not that helpful anyway
    # you have to use bigrams if you use trigrams. also with use_multi, you can
    # TODO: if you have time try to make this work for "skip-gram". maybe write another method
    # TODO: that calculates skip-grams and adds them after the initial featurize method is over.
    if use_single_n == 0:
        if use_bigrams:
            train_dict_list = featurize_multi_type(train_docs, counts, top_100, use_trigrams)
            dev_dict_list = featurize_multi_type(dev_docs, counts, top_100, use_trigrams)
            test_dict_list = featurize_multi_type(test_docs, counts, top_100, use_trigrams)
        else:
            train_dict_list = featurize_single_type(train_docs, counts, 1)
            dev_dict_list = featurize_single_type(dev_docs, counts, 1)
            test_dict_list = featurize_single_type(test_docs, counts, 1)
    else:
        if use_single_n == 1:
            train_dict_list = featurize_single_type(train_docs, counts, 1)
            dev_dict_list = featurize_single_type(dev_docs, counts, 1)
            test_dict_list = featurize_single_type(test_docs, counts, 1)
        elif use_single_n == 2:
            train_dict_list = featurize_single_type(train_docs, counts, 2)
            dev_dict_list = featurize_single_type(dev_docs, counts, 2)
            test_dict_list = featurize_single_type(test_docs, counts, 2)
        elif use_single_n == 3:
            train_dict_list = featurize_single_type(train_docs, counts, 3)
            dev_dict_list = featurize_single_type(dev_docs, counts, 3)
            test_dict_list = featurize_single_type(test_docs, counts, 3)

    # create vectorizer and vectorize each set
    vectorizer = sklearn.feature_extraction.DictVectorizer()
    feature_vec_train = vectorizer.fit_transform(train_dict_list)
    feature_vec_dev = vectorizer.transform(dev_dict_list)
    feature_vec_test = vectorizer.transform(test_dict_list)
    model = sklearn.naive_bayes.MultinomialNB(alpha= alpha)
    # train the model
    model.fit(feature_vec_train, labels_train)
    # predict on dev set or test set depending on original parameter.
    if not use_test:
        return model.predict(feature_vec_dev)
    else:
        return model.predict(feature_vec_test)


# would rather add another method for logistic model predictions than add more parameters to these.
def logistic_train_predict(big_c, train_docs, dev_docs, test_docs, labels_train, counts, use_bigrams, use_trigrams, top_100, use_single_n: int = 0, use_test: bool = False):
    # bigrams are used by default in featurize_multi_type, and since they're not that helpful anyway
    # you have to use bigrams if you use trigrams. also with use_multi, you can
    # TODO: if you have time try to make this work for "skip-gram". maybe write another method
    # TODO: that calculates skip-grams and adds them after the initial featurize method is over.
    if use_single_n == 0:
        if use_bigrams:
            train_dict_list = featurize_multi_type(train_docs, counts, top_100, use_trigrams)
            dev_dict_list = featurize_multi_type(dev_docs, counts, top_100, use_trigrams)
            test_dict_list = featurize_multi_type(test_docs, counts, top_100, use_trigrams)
        else:
            train_dict_list = featurize_single_type(train_docs, counts, 1)
            dev_dict_list = featurize_single_type(dev_docs, counts, 1)
            test_dict_list = featurize_single_type(test_docs, counts, 1)
    else:
        if use_single_n == 1:
            train_dict_list = featurize_single_type(train_docs, counts, 1)
            dev_dict_list = featurize_single_type(dev_docs, counts, 1)
            test_dict_list = featurize_single_type(test_docs, counts, 1)
        elif use_single_n == 2:
            train_dict_list = featurize_single_type(train_docs, counts, 2)
            dev_dict_list = featurize_single_type(dev_docs, counts, 2)
            test_dict_list = featurize_single_type(test_docs, counts, 2)
        elif use_single_n == 3:
            train_dict_list = featurize_single_type(train_docs, counts, 3)
            dev_dict_list = featurize_single_type(dev_docs, counts, 3)
            test_dict_list = featurize_single_type(test_docs, counts, 3)

    # create vectorizer and vectorize each set
    vectorizer = sklearn.feature_extraction.DictVectorizer()
    feature_vec_train = vectorizer.fit_transform(train_dict_list)
    feature_vec_dev = vectorizer.transform(dev_dict_list)
    feature_vec_test = vectorizer.transform(test_dict_list)
    model = LogisticRegression(C=big_c, max_iter=500)
    # train the model
    model.fit(feature_vec_train, labels_train)
    # predict on dev set or test set depending on parameter
    if not use_test:
        return model.predict(feature_vec_dev)
    else:
        return model.predict(feature_vec_test)


def add_tokenized_col(data: pandas.DataFrame):
    data["Tokens"] = data["Line"].apply(lambda line: nltk.tokenize.word_tokenize(line))
    return data


def add_ngram_feature_cols(data: pandas.DataFrame):
    data = add_tokenized_col(data)
    data["Bigrams"] = data["Tokens"].apply(lambda token_list: get_ngrams(token_list, 2))
    data["Trigrams"] = data["Tokens"].apply(lambda token_list: get_ngrams(token_list, 3))
    return data


def featurize_single_type(data, counts: bool, n: int, top100:bool = False):
    """
    this gets the list of feature dictionaries to use with DictVectorizer.
    works up to trigrams with counts and binary.
    """
    feature_dict_list = []
    # so far this is for unigrams
    if not top100:
        if n == 1:
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
        elif n == 2:
            if not counts:
                for line in data["Bigrams"]:
                    line_dict = dict()
                    for token in line:
                        line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            else:
                for line in data["Bigrams"]:
                    line_dict = dict()
                    for token in line:
                        if token in line_dict:
                            line_dict[token] += 1
                        else:
                            line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            return feature_dict_list
        elif n == 3:
            if not counts:
                for line in data["Trigrams"]:
                    line_dict = dict()
                    for token in line:
                        line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            else:
                for line in data["Trigrams"]:
                    line_dict = dict()
                    for token in line:
                        if token in line_dict:
                            line_dict[token] += 1
                        else:
                            line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            return feature_dict_list
        # note no else here for n sizes because i'm only ever considering up to trigrams
    else:
        # here we do everything again but with the top 100 by frequency for each ngram
        if n == 1:
            if not counts:
                for line in data["Top_Unigrams"]:
                    line_dict = dict()
                    for token in line:
                        line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            else:
                for line in data["Top_Unigrams"]:
                    line_dict = dict()
                    for token in line:
                        if token in line_dict:
                            line_dict[token] += 1
                        else:
                            line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            return feature_dict_list
        elif n == 2:
            if not counts:
                for line in data["Top_Bigrams"]:
                    line_dict = dict()
                    for token in line:
                        line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            else:
                for line in data["Top_Bigrams"]:
                    line_dict = dict()
                    for token in line:
                        if token in line_dict:
                            line_dict[token] += 1
                        else:
                            line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            return feature_dict_list
        elif n == 3:
            if not counts:
                for line in data["Top_Trigrams"]:
                    line_dict = dict()
                    for token in line:
                        line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            else:
                for line in data["Top_Trigrams"]:
                    line_dict = dict()
                    for token in line:
                        if token in line_dict:
                            line_dict[token] += 1
                        else:
                            line_dict[token] = 1
                    feature_dict_list.append(line_dict)
            return feature_dict_list


# this method combines unigrams with bigrams and optionally with trigrams. also parameters for counts and
# if you're using top 100 for features other than unigrams


def featurize_multi_type(data, counts: bool, other_feats_top100:bool = True, use_trigrams:bool = False):
    unigram_dicts = featurize_single_type(data, counts, 1)
    combined_dict_list = []
    if other_feats_top100:
        if use_trigrams:
            bigram_dicts = featurize_single_type(data, counts, 2, True)
            trigram_dicts = featurize_single_type(data, counts, 3, True)
            for i in range(len(bigram_dicts)):
                combined = unigram_dicts[i] | bigram_dicts[i] | trigram_dicts[i]
                combined_dict_list.append(combined)
            return combined_dict_list
        else:
            bigram_dicts = featurize_single_type(data, counts, 2, True)
            for i in range(len(bigram_dicts)):
                combined = unigram_dicts[i] | bigram_dicts[i]
                combined_dict_list.append(combined)
            return combined_dict_list
    else:
        if use_trigrams:
            bigram_dicts = featurize_single_type(data, counts, 2)
            trigram_dicts = featurize_single_type(data, counts, 3)
            for i in range(len(bigram_dicts)):
                combined = unigram_dicts[i] | bigram_dicts[i] | trigram_dicts[i]
                combined_dict_list.append(combined)
            return combined_dict_list
        else:
            bigram_dicts = featurize_single_type(data, counts, 2)
            for i in range(len(bigram_dicts)):
                combined = unigram_dicts[i] | bigram_dicts[i]
                combined_dict_list.append(combined)
            return combined_dict_list


# MUST BE RUN BEFORE FEATURIZING IF YOU WANT TOP 100

def find_top_100(data, n):
    """Gets the top 100 (or less) ngrams by count and adds a column in the dataframe for it."""
    counts = ngram_counts(data, n)
    if n == 1:
        top_unigrams = {key for key, count in counts.most_common(min(len(counts),100))}
        top_list = []
        for row in data["Tokens"]:
            row_list = [token for token in row if token in top_unigrams]
            top_list.append(row_list)
        data["Top_Unigrams"] = top_list
        return data
    elif n == 2:
        top_bigrams = {key for key, count in counts.most_common(min(len(counts),100))}
        top_list = []
        for row in data["Bigrams"]:
            row_list = [token for token in row if token in top_bigrams]
            top_list.append(row_list)
        data["Top_Bigrams"] = top_list
        return data
    elif n == 3:
        top_bigrams = {key for key, count in counts.most_common(min(len(counts),100))}
        top_list = []
        for row in data["Trigrams"]:
            row_list = [token for token in row if token in top_bigrams]
            top_list.append(row_list)
        data["Top_Trigrams"] = top_list
        return data



def get_ngrams(tokens, n):
    ngram_list = [str(ngram) for ngram in ngrams(tokens, n)]
    return ngram_list


# keeps counts for ngrams seen in all data. likely affected by whether features are binary or counts
# so far works for 1-3grams


def ngram_counts(data, n):
    if n == 1:
        all_unigrams = []
        for uni_list in data["Tokens"]:
            all_unigrams.extend(uni_list)
        unigram_counts = Counter(all_unigrams)
        return unigram_counts
    elif n == 2:
        all_bigrams = []
        for bi_list in data["Bigrams"]:
            all_bigrams.extend(bi_list)
        bigram_counts = Counter(all_bigrams)
        return bigram_counts
    elif n == 3:
        all_trigrams = []
        for tri_list in data["Trigrams"]:
            all_trigrams.extend(tri_list)
        trigram_counts = Counter(all_trigrams)
        return trigram_counts


if __name__ == '__main__':
    main()
