from classes.leaf import get_best_leaf_probs
from classes.tree import build_tree, classify_tree, print_tree

from classes.functions import load_and_split

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':



    # # Specify the header. The output/target is always the last column of the header!
    # header = ['Glucose', 'BMI', 'Pregnancies', 'DiabetesPedigreeFunction', 'Outcome']
    #
    # # Split train and test after loading the CSV file
    # train, test = load_and_split('diabetes.csv', header)

    # Specify the header. The output/target is always the last column of the header!
    header = ['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'thal', 'target']

    # Split train and test after loading the CSV file
    train, test = load_and_split('heart.csv', header)

    # Build the tree with the training data
    my_tree = build_tree(train)

    # Print the tree
    print_tree(header, my_tree)

    # Initialize the dictionary for the confusion matrix
    conf_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, }

    # Predict the output for each test row
    for row in test:
        test_value = row[-1]

        pred_value = get_best_leaf_probs(classify_tree(row, my_tree))

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

    # Compute the metrics score and print them
    accuracy = (conf_matrix['TP'] + conf_matrix['TN']) / sum(conf_matrix.values())
    precision = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    recall = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    f1_score = 2 * ((precision*recall)/(precision+recall))

    print("\naccuracy : {}".format(round(accuracy, 2)))
    print("precision : {}".format(round(precision, 2)))
    print("recall : {}".format(round(recall, 2)))
    print("f1_score : {}".format(round(f1_score, 2)))

    # Plot the confusion matrix
    ax = sns.heatmap([[conf_matrix['TN'], conf_matrix['FN']], [conf_matrix['FP'], conf_matrix['TP']]],
                     annot=True,
                     fmt='d',
                     cbar=False,
                     cmap=ListedColormap(['#FB8875', '#A8D08D']))
    ax.set(xlabel='True value', ylabel='Predicted value')
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()


