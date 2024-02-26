from hyperedge import HyperedgeType
from ..rbf.gaussian import gaussian_expansion
from ..neighbor_list import get_nbrlist


### Define bonds hyperedge type for generation
class Bonds(HyperedgeType):
    def __init__(self, dir_or_nbrset=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'bond'
        self.order = 2

        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)
    
    def generate(self, dir_or_nbrset, nn_strat = 'voro', gauss_dim = 40, radius = 8):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else: 
            nbr_list = dir_or_nbrset
        self.nbrset = nbr_list
        self.nbr_strategy = nn_strat
            
        if gauss_dim != 1:
            ge = gaussian_expansion(dmin = 0, dmax = radius, steps = gauss_dim)
            
        distances = []
        bond_index = 0
        ## currently double counts pair-wise edges/makes undirected edges
        for neighbor_set in nbr_list:
            center_index = neighbor_set[0]
            for neighbor in neighbor_set[1]:
                neigh_index = neighbor[0]
                offset = neighbor[1]
                distance = neighbor[2]
            
                self.hyperedge_index[0].append(center_index)
                self.hyperedge_index[1].append(bond_index)

                self.hyperedge_index[0].append(neighbor[0])
                self.hyperedge_index[1].append(bond_index)
            
                self.neighborsets.append([center_index,neighbor[0]])
                
                distances.append(distance)
            
                bond_index += 1

            
        if self.generate_features:
            for dist in distances:
                if gauss_dim != 1:
                    dist = ge.expand(dist)
                self.hyperedge_attrs.append(dist)
