from classes.functions import class_counts


class Leaf:
    """A Leaf node classifies data.
    It holds a dict of the labels and their numbers reaching this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


def get_best_leaf_probs(counts):
    """A way to get the best probability in the leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = int(counts[lbl] / total * 100)

    return max(probs, key=probs.get)
