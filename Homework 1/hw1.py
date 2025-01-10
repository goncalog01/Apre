import scipy.io.arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import ttest_ind

# ficheiro sem observacoes com entradas "?"
FILE_NAME, SEED, genmat = "breast.w.new.arff", 17, lambda:[[0 for i in range(10)] for i in range(9)]

data, meta = scipy.io.arff.loadarff(FILE_NAME)

#question 5
benign_m = genmat()
malign_m = genmat()

for entry in data:
    if entry['Class'] == b'benign':
        matrix = benign_m
    else:
        matrix = malign_m
    for i in range(9):
        matrix[i][int(entry[i]) - 1] += 1

fig, ax = plt.subplots(3,3, sharex=True, sharey=True)

for n in range(9):
    ax[n // 3, n % 3].bar([el for el in range(1,11)], benign_m[n])
    ax[n // 3, n % 3].bar([el for el in range(1,11)], malign_m[n])

plt.show()

#questions 6/7
knn_classifiers = dict()

# separate attributes from classes
targets = list(el["Class"] for el in data)
training_data = np.array(list(list(el[i] for i in range(9)) for el in data))
k_fold = KFold(n_splits=10, shuffle=True, random_state=SEED)

for n in (3, 5, 7):
    knn_classifiers[n] = KNeighborsClassifier(n_neighbors=n, p=2, weights="uniform")

naive_bayes_classifier, folds, nb_folds, i = MultinomialNB(), [], [], 1
for train, test in k_fold.split(training_data):
    for n in (3, 5, 7):
        knn_classifiers[n].fit(np.array([training_data[i] for i in train]), np.array([targets[i] for i in train]))
        acc = knn_classifiers[n].score(np.array([training_data[i] for i in test]), np.array([targets[i] for i in test]), sample_weight=None)
        folds.append({ "fold" : i, "n" : n, "acc" : acc })

    naive_bayes_classifier.fit(np.array([training_data[i] for i in train]), np.array([targets[i] for i in train]))
    nb_acc = naive_bayes_classifier.score(np.array([training_data[i] for i in test]), np.array([targets[i] for i in test]), sample_weight=None)
    nb_folds.append({ "fold" : i, "acc" : nb_acc})
    i += 1

accs = { 3 : [], 5 : [], 7 : [] }
nb_accs = []
avg_accs = { 3 : 0, 5 : 0, 7 : 0 }
avg_nb_acc = 0

for fold in folds:
    accs[fold['n']].append(fold['acc'])

for nb_fold in nb_folds:
    nb_accs.append(nb_fold["acc"])

for n in accs:
    avg_accs[n] = sum(accs[n]) / 10

avg_nb_acc = sum(nb_accs) / 10
knn_3_acc = accs[3]
statistic, pvalue = ttest_ind(knn_3_acc, nb_accs, alternative="less") #hypothesis test