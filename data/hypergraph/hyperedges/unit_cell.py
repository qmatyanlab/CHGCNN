
from .hyperedge import HyperedgeType
from ..rbf.gaussian import gaussian_expansion
from ..neighbor_list import get_nbrlist


### Define bonds hyperedge type for generation
class UnitCell(HyperedgeType):
    def __init__(self, dir_or_struc=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'unit_cell'
        self.neighborsets = [[]]
        self.order = 100

        if dir_or_struc!=None:
            self.generate(dir_or_struc)

    def generate(self, dir_or_struc):
        if type(dir_or_struc) == str:
            struc = CifParser(dir_or_struc).get_structures()[0]
        else: 
            struc = dir_or_struc
        
        
        for site_index in range(len(struc.sites)):
            self.hyperedge_index[0].append(site_index)
            self.hyperedge_index[1].append(0)
            
            self.neighborsets[0].append(site_index)
            
            
        if self.generate_features:
            structure_fingerprint = get_structure_fingerprint(struc)
            self.hyperedge_attrs.append(structure_fingerprint)
        
