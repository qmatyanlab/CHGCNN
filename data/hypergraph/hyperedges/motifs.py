from .hyperedge import HyperedgeType
from ..rbf.gaussian import gaussian_expansion

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

from matminer.featurizers.site.fingerprint import OPSiteFingerprint, ChemEnvSiteFingerprint
from ..neighbor_list import get_nbrlist

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.local_env import LocalStructOrderParams


import numpy as np
import math


### Define bonds hyperedge type for generation
class Motifs(HyperedgeType):
    def __init__(self,  dir_or_nbrset=None, struc=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'motif'
        self.order = 12
        self.struc=struc
        
        self.all_lsop_types = [ "cn",
                            "sgl_bd",
                            "bent",
                            "tri_plan",
                            "tri_plan_max",
                            "reg_tri",
                            "sq_plan",
                            "sq_plan_max",
                            "pent_plan",
                            "pent_plan_max",
                            "sq",
                            "tet",
                            "tet_max",
                            "tri_pyr",
                            "sq_pyr",
                            "sq_pyr_legacy",
                            "tri_bipyr",
                            "sq_bipyr",
                            "oct",
                            "oct_legacy",
                            "pent_pyr",
                            "hex_pyr",
                            "pent_bipyr",
                            "hex_bipyr",
                            "T",
                            "cuboct",
                            "cuboct_max",
                            "see_saw_rect",
                            "bcc",
                            "q2",
                            "q4",
                            "q6",
                            "oct_max",
                            "hex_plan_max",
                            "sq_face_cap_trig_pris"]
        
        #Removed S:10, S:12, SH:11, CO:11 and H:10 due to errors in package
        self.all_ce_types = ['S:1', 
                             'L:2', 
                             'A:2', 
                             'TL:3', 
                             'TY:3', 
                             'TS:3', 
                             'T:4', 
                             'S:4', 
                             'SY:4', 
                             'SS:4', 
                             'PP:5', 
                             'S:5', 
                             'T:5', 
                             'O:6', 
                             'T:6', 
                             'PP:6', 
                             'PB:7', 
                             'ST:7', 
                             'ET:7', 'FO:7', 'C:8', 'SA:8', 'SBT:8', 'TBT:8', 'DD:8', 'DDPN:8', 'HB:8', 'BO_1:8', 'BO_2:8', 'BO_3:8', 'TC:9', 'TT_1:9', 'TT_2:9', 'TT_3:9', 'HD:9', 'TI:9', 'SMA:9', 'SS:9', 'TO_1:9', 'TO_2:9', 'TO_3:9', 'PP:10', 'PA:10', 'SBSA:10', 'MI:10', 'BS_1:10', 'BS_2:10', 'TBSA:10', 'PCPA:11', 'H:11', 'DI:11', 'I:12', 'PBP:12', 'TT:12', 'C:12', 'AC:12', 'SC:12', 'HP:12', 'HA:12']
        
        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)

    def generate(self, dir_or_nbrset, nn_strat = 'mind', lsop_types = [], ce_types = []):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else: 
            nbr_list = dir_or_nbrset 
            if self.struc == None:
                print('Structure required as input for motif neighbor lists')
            struc = self.struc

        self.nbr_strategy = nn_strat

        neighborhoods = []
        motif_index = 0
        for n, neighborset in nbr_list:
            neigh_idxs = []
            for idx in neighborset:
                neigh_idx = idx[0]
                neigh_idxs.append(neigh_idx)
                self.hyperedge_index[0].append(neigh_idx)
                self.hyperedge_index[1].append(motif_index)
            self.hyperedge_index[0].append(n)
            self.hyperedge_index[1].append(motif_index)
            neighborhoods.append([n, neigh_idxs])
            neigh_idxs.append(n)
            self.neighborsets.append(neigh_idxs)
            motif_index += 1
        if self.generate_features == True and lsop_types == []:
            lsop_types = self.all_lsop_types
        if self.generate_features == True and ce_types == []:
            ce_types = self.all_ce_types

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(
            centering_type="centroid",
            include_central_site_in_centroid=True,
            structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE,
        )

        ##Compute order parameter features
        lsop = LocalStructOrderParams(lsop_types)
        CSM = ChemEnvSiteFingerprint(ce_types, MultiWeightsChemenvStrategy.stats_article_weights_parameters(), lgf)

        lsop_tol = 0.05
        for site, neighs in neighborhoods:
            op_feat = lsop.get_order_parameters(struc, site, indices_neighs = neighs)
            csm_feat = CSM.featurize(struc, site)
            for n,f in enumerate(op_feat):
                if f == None:
                    op_feat[n] = 0
                elif f > 1:
                    op_feat[n] = f
                ##Account for tolerance:
                elif f > lsop_tol:
                    op_feat[n] = f
                else:
                    op_feat[n] = 0
            feat = np.concatenate((op_feat, csm_feat))
            self.hyperedge_attrs.append(feat)

