
### Define general hyperedge type class
class HyperedgeType(object):
    def __init__(self, generate_features = True):
        self.hyperedge_index = [[],[]]
        self.hyperedge_attrs = []
        self.neighborsets = []
        self.generate_features = generate_features
