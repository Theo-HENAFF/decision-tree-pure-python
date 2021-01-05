from classes.question import Question
from classes.index import Index


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def partition(rows, question):
    """Partitions a given dataset.
    For each row of the dataset, check whether the answer is True or False.
    Partition by this answer.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def find_best_split(rows, index="gini"):
    """Find the best question to ask by iterating over every feature/value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep track of the feature / value that produced it
    current_uncertainty = getattr(Index, index)(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):
        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain > best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


def info_gain(left, right, current_uncertainty, index="gini"):
    """Calculate the information Gain.
    The uncertainty of the starting node, minus the weighted impurity of thetwo child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * getattr(Index, index)(left) - (1 - p) * getattr(Index, index)(right)


def load_and_split(csv_name, column, train_part=0.8):
    """Load the CSV with pandas
    Select the column to keep
    Shuffle the dataset and split it into train and test dataset
    return train and test as python's nested list.
    """

    import pandas as pd

    data = pd.read_csv(csv_name)

    # Keep only selected columns and shuffle rows
    data = data[column].sample(frac=1, random_state=1)

    # Split train and test data
    rows_train_test_split = int(train_part * len(data))

    train = data.head(rows_train_test_split)
    test = data.tail(len(data) - rows_train_test_split)

    return train.values.tolist(), test.values.tolist()
