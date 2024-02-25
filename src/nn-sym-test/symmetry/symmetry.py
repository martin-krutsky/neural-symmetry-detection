from abc import ABC


class Symmetry(ABC):
    def __init__(self):
        pass

    def expand_equivalence_class(self, data_point):
        pass

    def filter_equivalence_class(self, array, data_point):
        pass
