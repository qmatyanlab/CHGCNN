from hyperedge import HyperedgeType
from ..rbf.gaussian import gaussian_expansion
from ..neighbor_list import get_nbrlist


### Define triplets hyperedge type for generation
class Triplets(HyperedgeType):
    def __init__(self, dir_or_nbrset=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'triplet'
        self.order = 3

        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)
    
    def generate(self, dir_or_nbrset, nn_strat = 'voro', gauss_dim = 40, radius = 8):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else:
            nbr_list = dir_or_nbrset
        if gauss_dim != 1:
            ge = gaussian_expansion(dmin = -1, dmax = 1, steps = gauss_dim)

        triplet_index = 0
        for cnt_idx, neighborset in nbr_list:
                for i in itertools.combinations(neighborset, 2):
                    (pair_1_idx, offset_1, distance_1), (pair_2_idx, offset_2, distance_2) = i

                    if self.generate_features == True:
                        offset_1 = np.stack(offset_1)
                        offset_2 = np.stack(offset_2)
                        cos_angle = (offset_1 * offset_2).sum(-1) / (np.linalg.norm(offset_1, axis=-1) * np.linalg.norm(offset_2, axis=-1))

                        #Stop-gap to fix nans from zero displacement vectors
                        cos_angle = np.nan_to_num(cos_angle, nan=1)

                        if gauss_dim != 1:
                            cos_angle = ge.expand(cos_angle)
                        
                        self.hyperedge_attrs.append(cos_angle)
                            
                    self.hyperedge_index[0].append(pair_1_idx)
                    self.hyperedge_index[1].append(triplet_index)

                    self.hyperedge_index[0].append(pair_2_idx)
                    self.hyperedge_index[1].append(triplet_index)
                    
                    self.hyperedge_index[0].append(cnt_idx)
                    self.hyperedge_index[1].append(triplet_index)
            
                    self.neighborsets.append([cnt_idx, pair_1_idx, pair_2_idx])
                    
                    triplet_index += 1

