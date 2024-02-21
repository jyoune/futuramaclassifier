# Futurama- Who Said It?
## Description
This classification project explores the task of matching character labels to lines of dialogue from a TV show. I obtained a [dataset](https://www.kaggle.com/datasets/josephvm/futurama-seasons-16-transcripts/data) of the lines of dialogue from the show Futurama, trimmed it to include the lines from the 7 most frequent characters (enumerated below), and trained various model configurations using scikit-learn on the set of lines to predict which character it came from.

## Data
After truncating the set to only the lines spoken by the 7 main characters (i.e the ones with the most lines), I had a total of 14,439 lines. The breakdown per character label is as follows:
| Character        | # Lines   |
|--------------|-----------|
| Fry | 3,791  |
| Bender      |3,395 |
| Leela | 2,998     |
| Farnsworth     | 1,730 |
| Zoidberg | 888     |
| Amy      | 863  |
| Hermes | 771     |

These lines were then split into train, dev, and test sets.


## Files

What follows is a brief description of each file inside this repository and their purpose. Note that the output in output.txt is an intermediate that is used to write the report in final_results.pdf

train_dev_test_split.py - takes the source file "only_spoken_text.csv", creates main_char_lines.csv and splits it into train, dev, and test, storing in csv format.

baseline.py- creates a baseline and outputs it to output.txt. Some code duplicated in main.py,
some is changed

main.py- takes in train, dev, test sets in csv and turns them into dataframes. 
extracts features and puts in those dataframes. has methods for running baseline, 
doing dev set predictions / grid searching (and outputting results to output.txt), 
as well as running best models on test and reporting results. also all other methods for featurizing and vectorizing,
training different models

output.txt- sets of intermediate outputs (feature combos with best parameters) with accuracies.

**final_results.pdf**- a writeup containing an overview and discussion of the project, as well as neatly tabulated results for the tested configurations, and a selection of the best-performing ones.

episode_list.csv- one of two source data files. has all lines plus non-spoken text.

only_spoken_text.csv- other of two source files. only spoken lines.

main_char_lines.csv- all lines where character label is one of main 7. only has line and label

training_set, development_set, test_set.csv- splits of main_char_lines

