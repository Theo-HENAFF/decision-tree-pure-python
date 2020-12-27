from classes.leaf import get_best_leaf_probs
from classes.tree import build_tree, classify_tree, print_tree

from classes.functions import load_and_split

if __name__ == '__main__':

    # The label is always the last column !
    header = ['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'thal', 'target']

    # split train and test
    train, test = load_and_split('heart.csv', header)

    my_tree = build_tree(train)

    print_tree(header, my_tree)

    good_pred = []
    for row in test:
        test_value = row[-1]

        pred_value = get_best_leaf_probs(classify_tree(row, my_tree))

        # Uncomment the following line to see each prediction
        # print("Actual: {}. Predicted: {}".format(test_value, pred_value))

        good_pred.append(test_value == pred_value)

    avg_score = sum(good_pred) / len(good_pred)
    print("average_accuracy : {}".format(avg_score))
