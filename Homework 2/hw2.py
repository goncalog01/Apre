import scipy.io.arff
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy

FILE_NAME = "breast.w.new.arff" # ficheiro sem observacoes com entradas "?"
arff = scipy.io.arff
SEED = 17
data, meta = arff.loadarff(FILE_NAME)
k_fold = KFold(n_splits=10, shuffle=True, random_state=SEED)
max_features = { 1 : None, 3 : None, 5 : None, 9 : None }
max_features_folds = list()
test_max_f_accs = list(0 for i in range(4))
train_max_f_accs = list(0 for i in range(4))
test_max_d_accs = list(0 for i in range(4))
train_max_d_accs = list(0 for i in range(4))
max_depth_folds = list()
max_depth = deepcopy(max_features)
kbests = list()
targets = list(el["Class"] for el in data)
training_data = np.array(list(list(el[i] for i in range(9)) for el in data))

for n in (1,3,5,9):
    kbests.append(SelectKBest(mutual_info_classif, k=n).fit_transform(training_data, targets))
    max_features[n] = DecisionTreeClassifier()
    max_depth[n] = DecisionTreeClassifier(max_depth=n)
    
i = 0

for n in (1, 3, 5, 9):
    for train, test in k_fold.split(kbests[i]):
        max_features[n].fit(np.array([kbests[i][j] for j in train]),\
        np.array([targets[j] for j in train]))
        test_acc = max_features[n].score(np.array([kbests[i][j] for j in test]),\
        np.array([targets[j] for j in test]), sample_weight=None)
        train_acc = max_features[n].score(np.array([kbests[i][j] for j in train]),\
        np.array([targets[j] for j in train]))
        test_max_f_accs[i] += test_acc
        train_max_f_accs[i] += train_acc
    i += 1

i = 0

for n in (1, 3, 5, 9):
    for train, test in k_fold.split(train):
        max_depth[n].fit(np.array([training_data[i] for i in train]),\
        np.array([targets[i] for i in train]))
        test_acc = max_depth[n].score(np.array([training_data[i] for i in test]),\
        np.array([targets[i] for i in test]), sample_weight=None)
        train_acc = max_depth[n].score(np.array([training_data[i] for i in train]),\
        np.array([targets[i] for i in train]))
        test_max_d_accs[i] += test_acc
        train_max_d_accs[i] += train_acc
    i += 1

test_max_f_accs = list(el / 10 for el in test_max_f_accs)
train_max_f_accs = list(el / 10 for el in train_max_f_accs)
test_max_d_accs = list(el / 10 for el in test_max_d_accs)
train_max_d_accs = list(el / 10 for el in train_max_d_accs) 