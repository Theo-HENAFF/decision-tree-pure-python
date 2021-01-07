from classes.tree import build_tree, print_tree

from classes.main_functions import load, load_and_split, predict, k_folds

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':

    # Specify the header. The output/target is always the last column of the header!
    header = ['Glucose', 'BMI', 'Pregnancies', 'DiabetesPedigreeFunction', 'Outcome']

    # Uncomment the following 11 lines if you want to perform a single prediction without cross-validation
    # # Split train and test after loading the CSV file
    # train, test = load_and_split('diabetes.csv', header)
    #
    # # Build the tree with the training data
    # my_tree = build_tree(train)
    #
    # # Print the tree
    # # print_tree(header, my_tree)
    #
    # # Predict with the test data
    # conf_matrix = predict(test, my_tree)

    # Perform a k-folds cross validation
    conf_matrix = k_folds(10, load('diabetes.csv', header))

    # Compute the metrics score and print them
    accuracy = (conf_matrix['TP'] + conf_matrix['TN']) / sum(conf_matrix.values())
    precision = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    recall = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    f1_score = 2 * ((precision*recall)/(precision+recall))

    print("\naccuracy : {}".format(round(accuracy, 2)))
    print("precision : {}".format(round(precision, 2)))
    print("recall : {}".format(round(recall, 2)))
    print("f1_score : {}".format(round(f1_score, 2)))

    # -------------------------
    # Plot the confusion matrix
    # -------------------------

    # we need to use numpy to convert the mask into an numpy array, because seaborn does not allow nested list as mask
    import numpy as np
    mask = np.array([[False, True], [True, False]])
    matrix = [[conf_matrix['TN'], conf_matrix['FN']], [conf_matrix['FP'], conf_matrix['TP']]]
    vmin = min([conf_matrix['TN'], conf_matrix['FN'], conf_matrix['FP'], conf_matrix['TP']])
    vmax = max([conf_matrix['TN'], conf_matrix['FN'], conf_matrix['FP'], conf_matrix['TP']])

    green = ListedColormap(['#A8D08D'])
    red = ListedColormap(['#FB8875'])

    # Plotting the first diagonal
    ax = sns.heatmap(matrix,
                     annot=True, fmt='d', cbar=False, vmin=vmin, vmax=vmax,
                     mask=mask, cmap=green,
                     )

    # Plotting the second diagonal
    ax = sns.heatmap(matrix,
                     annot=True, fmt='d', cbar=False, vmin=vmin, vmax=vmax,
                     mask=~mask, cmap=red,
                     )

    ax.set(xlabel='True value', ylabel='Predicted value')
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.show()


