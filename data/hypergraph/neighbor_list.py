from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import \
    LocalStructOrderParams, \
    VoronoiNN, \
    CrystalNN, \
    JmolNN, \
    MinimumDistanceNN, \
    MinimumOKeeffeNN, \
    EconNN, \
    BrunnerNN_relative, \
    MinimumVIRENN

import numpy as np

#generate custom neighbor list to be used by all struc2's with nearest neighbor determination technique as parameter
def get_nbrlist(struc, nn_strategy = 'mind', max_nn=12):
    NN = {
        # these methods consider too many neighbors which may lead to unphysical resutls
        'voro': VoronoiNN(tol=0.2),
        'econ': EconNN(),
        'brunner': BrunnerNN_relative(),

        # these two methods will consider motifs center at anions
        'crys': CrystalNN(),
        'jmol': JmolNN(),

        # not sure
        'minokeeffe': MinimumOKeeffeNN(),

        # maybe the best
        'mind': MinimumDistanceNN(),
        'minv': MinimumVIRENN()
    }

    nn = NN[nn_strategy]

    center_idxs = []
    neighbor_idxs = []
    offsets = []
    distances = []

    reformat_nbr_lst = []

    for n in range(len(struc.sites)):
        neigh = []
        neigh = [neighbor for neighbor in nn.get_nn(struc, n)]

        neighbor_reformat=[]
        for neighbor in neigh[:max_nn]:
            neighbor_index = neighbor.index
            offset = struc.frac_coords[neighbor_index] - struc.frac_coords[n] + neighbor.image
            m = struc.lattice.matrix
            offset = offset @ m
            distance = np.linalg.norm(offset)
            center_idxs.append(n)
            neighbor_idxs.append(neighbor_index)
            offsets.append(offset)
            distances.append(distance)

            neighbor_reformat.append((neighbor_index, offset, distance))
        reformat_nbr_lst.append((n,neighbor_reformat))

    return reformat_nbr_lst, nn_strategy
