import numpy as np
import math

import os
import os.path as osp
import csv
import json
import itertools
import time

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure


import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from hypergraph.hypergraph import Crystal_Hypergraph

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import math




##Build data structure in form of (vanilla) pytorch dataset (not PytorchGeometric!)
class CrystalHypergraphDataset(Dataset):
    def __init__(self, cif_dir, dataset_ratio=1.0, radius=4.0, n_nbr=12):
        super().__init__()

        self.radius = radius
        self.n_nbr = n_nbr

        ##dataset_ratio for testing
        self.dataset_ratio = dataset_ratio

        self.cif_dir = cif_dir

        with open(osp.join(cif_dir,'id_band_form_hull_metalicity.csv')) as id_prop:
            id_prop = csv.reader(id_prop)
            self.id_prop_data = [row for row in id_prop]
        
        with open(osp.join(cif_dir,'idsym_symbol_num_point_lattice.csv'), mode="r", newline="") as id_syms:
            id_sym_props = csv.reader(id_syms, delimiter=",")
            self.sym_props = {id_sym[0]: [id_sym[1],id_sym[2],id_sym[3],id_sym[4]] for id_sym in id_sym_props}

    def __len__(self):
        return int(round(len(self.id_prop_data)*self.dataset_ratio,1))
    
    def __getitem__(self, index, report = True):
        mp_id, band_gap, form_en, en_hull, metalicity = self.id_prop_data[index]
        target_dict = {
            'band_gap': torch.tensor(float(band_gap)).float(),
            'formation_energy': torch.tensor(float(form_en)).float(),
            'energy_abv_hull': torch.tensor(float(en_hull)).float(),
            'metalicity': torch.tensor(int(metalicity)).long(),
        }

        symbol, number, point_group, lattice = self.sym_props[mp_id]
        crystal_path = osp.join(self.cif_dir, mp_id)
        crystal_path = crystal_path + '.cif'
        if report:
            start = time.time()
        struc = CifParser(crystal_path).get_structures()[0]
        hgraph = Crystal_Hypergraph(struc, mp_id=mp_id, target_dict=target_dict)
        hgraph.symmetry_number = torch.tensor(int(number)).long()
        if report:
            duration = time.time()-start
            print(f'Processed {mp_id} in {round(duration,5)} sec')
        return {
            'hgraph': hgraph,
            'mp_id' : mp_id
            }
    
class InMemoryCrystalHypergraphDataset(Dataset):
    def __init__(self, data_dir, csv_dir = ''):
        super().__init__()

        if csv_dir == '':
            csv_dir = data_dir

        self.csv_dir = csv_dir
        self.data_dir = data_dir

        with open(osp.join(csv_dir, 'processed_ids.csv')) as id_file:
            ids_csv = csv.reader(id_file)
            ids = [mp_id[0] for mp_id in ids_csv]
            self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        mp_id = self.ids[index]
        file_dir = osp.join(self.data_dir, mp_id + '_hg.pt')
        data = torch.load(file_dir)
        num_nodes = list(data['atom'].hyperedge_attrs.shape)[0]
        data['atom'].num_nodes = torch.tensor(num_nodes).long()
        data['atom'].batch = torch.tensor([0 for i in range(num_nodes)])

        return data
    
processed_data_dir = 'dataset'

def process_data(idx):
    with open(osp.join('dataset','processed_ids.csv'),'a') as ids:
        try:
            d = dataset[idx]
            torch.save(d['hgraph'], 'dataset/{}_hg.pt'.format(d['mp_id']))
            ids.write(d['mp_id']+'\n')

        except:
            print(f'Cannot process index {idx}')


def run_process(N=None, processes=10):
    if N is None:
        N = len(dataset)

    pool = Pool(processes)

    for _ in tqdm(pool.imap_unordered(process_data, range(N)), total=N):
        pass


if __name__ == '__main__':
    cif_dir = 'cifs'
    processed_data_dir = processed_data_dir

    dataset = CrystalHypergraphDataset(cif_dir)
    with open(osp.join('dataset','processed_ids.csv'),'w') as ids:
        print(f'Clearing id list in {osp.join(processed_data_dir,"processed_ids.csv")}')
    run_process()
