class Question:
    """A Question is used to partition a dataset.
    To have a question it just need a column name and a split value.
    So the class take a column index from the header and a split value
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the feature value in this question.
        val = example[self.column]
        if isinstance(val, int) or isinstance(val, float):  # Test if val is a numerical value
            return val >= self.value
        else:
            return val == self.value

    def print_question(self, header):
        # Help to print
        condition = "=="
        if isinstance(self.value, int) or isinstance(self.value, float):  # Test if value is numeric
            condition = ">="
        return "Is {} {} {}?".format(header[self.column], condition, str(self.value))
