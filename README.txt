train_dev_test_split.py - takes the source file "only_spoken_text.csv", creates main_char_lines.csv and splits it into train, dev, and test, storing in csv format.

baseline.py- creates a baseline and outputs it to output.txt. Some code duplicated in main.py,
some is changed

main.py- takes in train, dev, test sets in csv and turns them into dataframes. 
extracts features and puts in those dataframes. has methods for running baseline, 
doing dev set predictions / grid searching (and outputting results to output.txt), 
as well as running best models on test and reporting results. also all other methods for featurizing and vectorizing,
training different models

output.txt- sets of outputs (feature combos with best parameters) with accuracies etc.

episode_list.csv- one of two source data files. has all lines etc plus non-spoken

only_spoken_text.csv- other of two source files. only spoken lines.

main_char_lines.csv- all lines where character label is one of main 7. only has line and label

training_set, development_set, test_set.csv- splits of main_char_lines
