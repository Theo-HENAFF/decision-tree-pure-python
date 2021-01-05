from classes.node import Node
from classes.leaf import Leaf
from classes.functions import find_best_split, partition


def build_tree(rows):
    """Builds the tree recursively.
    """

    # Partition the dataset on each of the unique attribute

    # Get the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # End condition: no info gain, it's a leaf because we can't ask any question
    if gain == 0:
        return Leaf(rows)

    # If the gain is not null we can partition the dataset
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node to save the best question to ask at this point and the branches.
    return Node(question, true_branch, false_branch)


def classify_tree(row, node):
    """Predict an output given a row of input by going recursively trough the tree.
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify_tree(row, node.true_branch)
    else:
        return classify_tree(row, node.false_branch)


def print_tree(header, node, spacing=""):
    """Recursive tree printing function.
    """

    # End condition: the node is a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the node's question
    print(spacing + str(node.question.print_question(header)))

    # Go on the true branch
    print(spacing + '--> True:')
    print_tree(header, node.true_branch, spacing + "  ")

    # Go on the false branch
    print(spacing + '--> False:')
    print_tree(header, node.false_branch, spacing + "  ")
