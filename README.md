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



