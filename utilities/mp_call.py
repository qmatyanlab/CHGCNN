from mp_api import *
from mp_api.client import MPRester

from pymatgen.io.cif import CifWriter

import pickle
import csv

#Contact materials project with api key and download all material
#structures and material ids of crytalline structures (those with
#some defined crystal system)

#We also choose several target for regression:band_gap, formation_energy_per_atom, energy_above_hull;
#and for classification: is_metal
#fields = ['band_gap','formation_energy_per_atom', 'energy_above_hull','is_metal']

fields = ['band_gap', 'formation_energy_per_atom', 'energy_above_hull', 'material_id', 'structure']

print(f"Downloading crystalline structure fields: {fields}")
with MPRester("M3npHaL9jodYuqvW6q9faTs4koRBRB9G") as mpr:
    docs = mpr.summary.search(
        band_gap = (0, None), fields=fields)
print(f"Found {len(docs)} crystalline structures")


#Print .cif files of all downloaded structures via CifWriter and
#associated write file property

direc = 'cifs'

for crys in docs:
    crys_cif=CifWriter(crys.structure, significant_figures=4)
    crys_cif.write_file(f'{direc}/{crys.material_id}.cif')
    print(f'Generated cif file for {crys.material_id}')
print(f"\n\nCifs stored in {direc}")

#Create list of material_ids for further property searches

        #Apparently the materials project partitions different
        #types of material properties into different databases.
        #Hence, we may need a different call for every database



with open(f"{direc}/id_prop_band_form_hull.csv", mode="w", newline="") as id_prop:
    id_prop = csv.writer(id_prop, delimiter=",")
    for crys in docs:
        id_prop.writerow([crys.material_id,crys.band_gap, crys.formation_energy_per_atom, crys.energy_above_hull])

print(f"Saved data to csv {direc}/id_prop_band_form_hull.csv")
