# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../var/folders/2g/m14hqhp55xj20c6g98wmncww0000gn/T'))
	print(os.getcwd())
except:
	pass
# %%
import pandas as pd


# %%
file = 'dataset_github/2019/main/labels.csv'
names = ['filename','standard','task2_class','tech_cond','Bathroom','Bathroom cabinet','Bathroom sink','Bathtub','Bed','Bed frame','Bed sheet','Bedroom','Cabinetry','Ceiling','Chair','Chandelier','Chest of drawers','Coffee table','Couch','Countertop','Cupboard','Curtain','Dining room','Door','Drawer','Facade','Fireplace','Floor','Furniture','Grass','Hardwood','House','Kitchen','Kitchen & dining room table','Kitchen stove','Living room','Mattress','Nightstand','Plumbing fixture','Property','Real estate','Refrigerator','Roof','Room','Rural area','Shower','Sink','Sky','Table','Tablecloth','Tap','Tile','Toilet','Tree','Urban area','Wall','Window']
labels_data = pd.read_csv(file, names=names)


# %%
labels_data = labels_data.drop(labels_data.index[0])


# %%
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# %%
for filename in labels_data['filename']:
    file_path = find(filename, "dataset_pendrive/")

    #do something with the file
    print(file_path)
    file = open(file_path, "r")
    file.close()

