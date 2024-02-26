from mp_api import *
from mp_api.client import MPRester

from pymatgen.io.cif import CifWriter

import os.path as osp

import pickle
import csv

#Contact materials project with api key and download all material
#structures and material ids of crytalline structures (those with
#some defined crystal system)

#We also choose several target for regression:band_gap, formation_energy_per_atom, energy_above_hull;
#and for classification: is_metal
#fields = ['band_gap','formation_energy_per_atom', 'energy_above_hull','is_metal']
processed_data_direc = 'dataset'

with open(osp.join(processed_data_direc, 'ids.csv'), mode="r", newline="") as id_prop:
    id_mats = csv.reader(id_prop)
    ids = [row[0] for row in id_mats]

print(ids)

fields = ['material_id', 'symmetry']

print(f"Downloading crystalline structure fields: {fields}")
with MPRester("M3npHaL9jodYuqvW6q9faTs4koRBRB9G") as mpr:
    docs = mpr.materials.search(material_ids = ids, fields=fields)
print(f"Found {len(docs)} crystalline structures")


with open(f"{direc}/idsym_symbol_num_point_lattice.csv", mode="w", newline="") as id_prop:
    id_prop = csv.writer(id_prop, delimiter=",")
    for crys in docs:
        id_prop.writerow([crys.material_id, crys.symmetry.symbol, crys.symmetry.number, crys.symmetry.point_group, crys.symmetry.crystal_system])
print(f"Saved symmetry data to idsym_symbol_num_point_lattice.csv")
