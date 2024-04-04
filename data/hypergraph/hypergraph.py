import numpy as np
import math

import os
import os.path as osp
import json
import itertools

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

import torch
from torch_geometric.data import HeteroData

from .hyperedges.bonds import Bonds
from .hyperedges.triplets import Triplets
from .hyperedges.motifs import Motifs
from .hyperedges.unit_cell import UnitCell

from .neighbor_list import get_nbrlist


### Define general crystal hypergraph class that accepts list of hyperedge types, mp_id string, and structure
class Crystal_Hypergraph(HeteroData):
    def __init__(self, struc = None, bonds = True, triplets = True, motifs = True, unit_cell = False,
                 mp_id: str = None, target_dict = {}, strategy = 'Aggregate'):
        super().__init__()  
        
        self.struc = struc
        self.mp_id = mp_id
        self.orders = []
        
        self.hyperedges = []
       
        if struc != None:
            ## Generate neighbor lists
            nbr_crys, _ = get_nbrlist(struc, nn_strategy = 'crys', max_nn=12)
            nbr_voro, _ = get_nbrlist(struc, nn_strategy = 'voro', max_nn=12)
        
            ## Generate bonds, triplets, motifs, and unit cell
            ## hyperedge types
            if bonds == True:
                bonds = Bonds(nbr_voro)
                self.hyperedges.append(bonds)
            if triplets == True:
                triplets = Triplets(nbr_voro)
                self.hyperedges.append(triplets)
            if motifs == True:
                motifs = Motifs(nbr_crys, struc=struc)    
                self.hyperedges.append(motifs)
            if unit_cell == True:
                unit_cell = UnitCell(struc)
                self.hyperedges.append(unit_cell)



            ## Add hyperedge types to hypergraph
            if self.hyperedges != None:
                for hyperedge_type in self.hyperedges:
                    self.add_hyperedge_type(hyperedge_type)
        
            ## Generate relatives edges and atomic info
            self.generate_atom_xs()
            self.generate_edges(strategy)
        
            ## Import target dict automatically, if passed as input of init
            if target_dict != {}:
                self.import_targets(target_dict)

    ## Function used to generate atomic features (Note these are considered hyperedge_attrs)
    def generate_atom_xs(self, import_feats=False):
        node_attrs = []
        for site in self.struc.sites:
            for el in site.species:
                node_attrs.append(el.Z)
    ## import features callsusual atom_init from CGCNN and assumes this json file 
    ## is in the current directory otherwise, feats are just atomic numbers
        if import_feats == True:
            with open('atom_init.json') as atom_init:
                atom_vecs = json.load(atom_init)
                node_attrs = [atom_vecs[f'{z}'] for z in node_attrs]
        self['atom'].hyperedge_attrs = torch.tensor(node_attrs).float()

    ## Function used to add hyperedge_type to hypergraph
    def add_hyperedge_type(self, hyperedge_type):
        self[('atom','in',hyperedge_type.name)].hyperedge_index = torch.tensor(hyperedge_type.hyperedge_index).long()
        self[(hyperedge_type.name,'contains','atom')].hyperedge_index = torch.flip(self[('atom','in',hyperedge_type.name)].hyperedge_index, dims=(0,)) 
        self[hyperedge_type.name].hyperedge_attrs = torch.tensor(np.stack(hyperedge_type.hyperedge_attrs)).float()
        self.orders.append(hyperedge_type.name)

    ## Function used to determine relatives edges between different order hyperedges
    def hyperedge_inclusion(self, larger_hedgetype, smaller_hedgetype, flip = False):
        hedge_index = [[],[]]
        for small_idx, small_set in enumerate(smaller_hedgetype.neighborsets):
            for large_idx, large_set in enumerate(larger_hedgetype.neighborsets):
                if contains(large_set, small_set):
                    hedge_index[0].append(small_idx)
                    hedge_index[1].append(large_idx)
        self[(smaller_hedgetype.name, 'in', larger_hedgetype.name)].hyperedge_index = torch.tensor(hedge_index).long()
        if flip == True:
            self[(larger_hedgetype.name, 'contains', smaller_hedgetype.name)].hyperedge_index = torch.tensor([hedge_index[1],hedge_index[0]]).long()

    ## Function used to determine relatives edges between touching hyperedges of same order
    def hyperedge_touching(self, hyperedge_type):
        hedge_index = [[],[]]
        for idx_1, set_1 in enumerate(hyperedge_type.neighborsets):
            for idx_2, set_2 in enumerate(hyperedge_type.neighborsets):
                if idx_1 == idx_2: 
                    pass
                else:
                    if touches(set_1, set_2):
                        hedge_index[0].append(idx_1)
                        hedge_index[1].append(idx_2)
        self[(hyperedge_type.name, 'touches', hyperedge_type.name)].hyperedge_index = torch.tensor(hedge_index).long()
      
    ## Function used to determine labelled inter-order hyperedge relations
    def hyperedge_relations(self, larger_hedgetype, smaller_hedgetype, flip = False):
        relation_index = [[],[],[]]
        for (idx_1, nset_1), (idx_2, nset_2) in itertools.combinations(enumerate(smaller_hedgetype.neighborsets),2):
            for large_idx, large_nset in enumerate(larger_hedgetype.neighborsets):
                if contains(large_nset, nset_1) and contains(large_nset, nset_2):
                    relation_index[0].append(idx_1)
                    relation_index[1].append(large_idx)
                    relation_index[2].append(idx_2)
                    if flip == True:
                        relation_index[0].append(idx_2)
                        relation_index[1].append(large_idx)
                        relation_index[2].append(idx_1)
        self[(smaller_hedgetype.name, larger_hedgetype.name, smaller_hedgetype.name)].inter_relations_index = torch.tensor(relation_index).long()


    ## Stop-gap for atom-wise relations, at the moment
    def atom_hyperedge_relations(self, larger_hedgetype, flip = False):
        relation_index = [[],[],[]]
        for idx, nset in enumerate(larger_hedgetype.neighborsets):
            for atom_pair in itertools.combinations(nset, 2):
                relation_index[0].append(atom_pair[0])
                relation_index[1].append(idx)
                relation_index[2].append(atom_pair[1])
                if flip == True:
                    relation_index[0].append(atom_pair[0])
                    relation_index[1].append(idx)
                    relation_index[2].append(atom_pair[1])
        self[('atom', larger_hedgetype.name, 'atom')].hyperedge_relations_index = torch.tensor(relation_index).long()

    ## Function used to genertate different edge strategies (Relatives, Aggregate, Interorder, All)
    def generate_edges(self, strategy):
        if strategy == 'All':
            self.generate_relatives()
        elif strategy == 'Relatives':
            self.generate_relatives(relatives = False)
        elif strategy == 'Aggregate': 
            self.generate_relatives(touching = False, relatives = False)
        elif strategy == 'Interorder': 
            self.generate_relatives(inclusion = False, touching = False)

        
    ## Function used to generate full relatives set
    def generate_relatives(self, relatives = True, touching = True, inclusion = True, flip = True):
        if relatives & inclusion == True:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_inclusion(pair_hedge_types[0],pair_hedge_types[1], flip = flip)
                        self.hyperedge_relations(pair_hedge_types[0],pair_hedge_types[1])
            for hedge_type in  self.hyperedges:
                self.atom_hyperedge_relations(hedge_type)

        
        elif inclusion:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_inclusion(pair_hedge_types[0],pair_hedge_types[1], flip = flip)
                        
        elif relatives:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_relations(pair_hedge_types[0],pair_hedge_types[1])
            for hedge_type in  self.hyperedges:
                self.atom_hyperedge_relations(hedge_type)

                
        if touching:
            for hyperedge_type in self.hyperedges:
                if hyperedge_type.name == 'unit_cell':
                    pass
                else:
                    self.hyperedge_touching(hyperedge_type)
        

    ## Import targets as dictionary and save as value of heterodata
    def import_targets(self, target_dict):
        for key, value in target_dict.items():
            self[key] = value



### Helper functions for inclusion and touching criteria of hyperedges
def contains(big, small):
    if all(item in big for item in small):
        return True
    else:
        return False
    
def touches(one, two):
    if any(item in one for item in two):
        return True
    else:
        return False
