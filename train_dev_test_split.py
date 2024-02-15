import pandas
import sklearn.model_selection


def main():
    all_docs = pandas.read_csv("only_spoken_text.csv", usecols=['Character', 'Line'])
    print(all_docs.loc[9, :].to_numpy().flatten().tolist())
    main_chars = ["Fry", "Leela", "Bender", "Farnsworth", "Amy", "Hermes", "Zoidberg"]
    main_docs = all_docs[all_docs["Character"].isin(main_chars)]
    main_docs = main_docs.reset_index(drop=True)
    print(main_docs.loc[9, :].to_numpy().flatten().tolist())
    # main_docs.to_csv('main_char_lines.csv', index=False)
    first_split = sklearn.model_selection.train_test_split(main_docs, test_size=0.20)
    for i in range(len(first_split)):
        first_split[i] = first_split[i].reset_index(drop=True)
    print(first_split[0].loc[0])
    print(first_split[1].loc[0])
    # i'm trying to rejoin the split line/char "series" in first_split by turning back into one column
    # dataframes then using the join method to put them back together into train, dev, and test
    # label-feature pairs
    train_docs = pandas.DataFrame(first_split[0])
    print(train_docs.loc[0])
    second_split = sklearn.model_selection.train_test_split(first_split[1], test_size=0.50)
    print(len(second_split))
    dev_docs = pandas.DataFrame(second_split[0])
    test_docs = pandas.DataFrame(second_split[1])
    print(len(train_docs), len(dev_docs), len(test_docs))
    train_docs = train_docs.reset_index(drop=True)
    dev_docs = dev_docs.reset_index(drop=True)
    test_docs = test_docs.reset_index(drop=True)
    print(train_docs.loc[0, :])
    print(dev_docs.loc[0, :])
    print(test_docs.loc[0, :])
    train_docs.to_csv('training_set.csv', index=False)
    dev_docs.to_csv('development_set.csv', index=False)
    test_docs.to_csv('test_set.csv', index=False)




if __name__ == '__main__':
    main()