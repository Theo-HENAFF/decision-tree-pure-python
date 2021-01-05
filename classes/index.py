def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Index:

    def gini(rows):
        """Calculate the Gini Impurity for a list of rows.
        There are a few different ways to do this, I thought this one was
        the most concise. See:
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        """
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity
