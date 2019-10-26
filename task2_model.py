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
# Read the dataset

file = 'dataset_github/2019/main/labels.csv'
names = ['filename','standard','task2_class','tech_cond','Bathroom','Bathroom cabinet','Bathroom sink','Bathtub','Bed','Bed frame','Bed sheet','Bedroom','Cabinetry','Ceiling','Chair','Chandelier','Chest of drawers','Coffee table','Couch','Countertop','Cupboard','Curtain','Dining room','Door','Drawer','Facade','Fireplace','Floor','Furniture','Grass','Hardwood','House','Kitchen','Kitchen & dining room table','Kitchen stove','Living room','Mattress','Nightstand','Plumbing fixture','Property','Real estate','Refrigerator','Roof','Room','Rural area','Shower','Sink','Sky','Table','Tablecloth','Tap','Tile','Toilet','Tree','Urban area','Wall','Window']
labels_data = pd.read_csv(file, names=names)


# %%
print(labels_data.shape)


# %%
labels_data.head(3)


# %%
# Clean the dataset

labels_data = labels_data.drop(labels=["filename", "standard", "tech_cond", "Bathroom", "Bedroom", "Living room", "Kitchen", "Dining room", "House"], axis=1)

labels_data = labels_data.drop(labels_data.index[0])

labels_data = labels_data[labels_data["task2_class"] !="validation"]


# %%
labels_data.head(3)


# %%
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# %%
# Prepare sets for training and validating models

array = labels_data.values

X = array[:,1:]
Y = array[:,0]

validation_size = 0.2
seed = 7
scoring = 'f1_weighted'

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# %%
# Test a few models and compare their results

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# %%
import matplotlib.pyplot as plt


# %%
# Compare model algorithms

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# %%
# Print data about the chosen model

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# %%
# Print a random prediction

print(model.predict(array[-2:-1,1:]))
print(array[-2:-1,0])


# %%
import pickle


# %%
# Save the model as file

model_filename = "task2_model.pkl"  

with open(model_filename, 'wb') as file:  
    pickle.dump(model, file)

