class Node:
    """A Node holds a question to ask to determine the true and the false branch. It keep also those branch."""

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
