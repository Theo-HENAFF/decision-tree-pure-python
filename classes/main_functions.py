from classes.leaf import get_best_leaf_probs
from classes.tree import classify_tree, build_tree
import pandas as pd


def load(csv_name, column):
    """Load the CSV with pandas
    Select the column to keep
    Shuffle the dataset and returns it as a nested list
    """

    data = pd.read_csv(csv_name)

    # Keep only selected columns and shuffle rows
    data = data[column].sample(frac=1, random_state=1)

    return data.values.tolist()


def load_and_split(csv_name, column, train_part=0.8):
    """Load the CSV with pandas
    Select the column to keep
    Shuffle the dataset and split it into train and test dataset
    return train and test as python's nested list.
    """

    data = pd.read_csv(csv_name)

    # Keep only selected columns and shuffle rows
    data = data[column].sample(frac=1, random_state=1)

    # Split train and test data
    rows_train_test_split = int(train_part * len(data))

    train = data.head(rows_train_test_split)
    test = data.tail(len(data) - rows_train_test_split)

    return train.values.tolist(), test.values.tolist()


def predict(test_data, tree):
    """Predict the output values given the test data and the tree built during training
    return the confusion matrix
    """

    # Initialize the dictionary for the confusion matrix
    conf_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, }

    # Predict the output for each test row
    for row in test_data:
        test_value = row[-1]

        pred_value = get_best_leaf_probs(classify_tree(row, tree))

        # Uncomment the following line to see each prediction
        # print("Actual: {}. Predicted: {}".format(test_value, pred_value))

        # Increment the right key/value of the confusion matrix
        if pred_value == 1.0 and test_value == 1.0:
            conf_matrix['TP'] += 1
        elif pred_value == 0.0 and test_value == 0.0:
            conf_matrix['TN'] += 1
        elif pred_value == 1.0 and test_value == 0.0:
            conf_matrix['FP'] += 1
        elif pred_value == 0.0 and test_value == 1.0:
            conf_matrix['FN'] += 1

    return conf_matrix


def k_folds(n_fold, data):
    """Do a K-Folds Cross Validation given a dataset (nested list)
    return the average confusion matrix as a python dict
    """

    # Define the length of the test dataset
    len_test = len(data) // n_fold

    # the list will keep a trace of all the prediction from each k-folds run
    list_conf = []

    for i in range(n_fold):
        print("K_fold : {}/{}".format(i+1, n_fold))

        test = data[i*len_test: (i+1) * len_test]

        if i == 0:
            train = data[(i+1)*len_test:]
        elif i == n_fold-1:
            train = data[: i * len_test]
        else:
            train = data[: i*len_test]+data[: (i+1) * len_test]

        # Build the tree with the train data and predict on the test data with it
        tree = build_tree(train)
        conf_matrix = predict(test, tree)

        list_conf.append([conf_matrix["TP"], conf_matrix["FP"], conf_matrix["TN"], conf_matrix["FN"]])

    sum_list_conf = [sum(i) for i in zip(*list_conf)]
    avg_conf_matrix = {"TP": int(round(sum_list_conf[0] / n_fold, 0)),
                       "FP": int(round(sum_list_conf[1] / n_fold, 0)),
                       "TN": int(round(sum_list_conf[2] / n_fold, 0)),
                       "FN": int(round(sum_list_conf[3] / n_fold, 0))
                       }
    return avg_conf_matrix

































