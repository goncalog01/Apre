import numpy as np
import scipy.io.arff
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #cross entropy
import matplotlib.pyplot as plt

arff = scipy.io.arff

breast_data, breast_meta = arff.loadarff("breast.w.new.arff")
kin_data, kin_meta = arff.loadarff("kin8nm.arff")

breast_targets = np.array(list(el["Class"] for el in breast_data))
breast_X = np.array(np.array(list(list(el[i] for i in range(9)) for el in breast_data)))

kin_targets = np.array(list(el["y"] for el in kin_data))
kin_X = np.array(np.array(list(list(el[i] for i in range(8)) for el in kin_data)))

#2
kfold = KFold(n_splits=5, random_state=0, shuffle=True)

classifier = MLPClassifier(hidden_layer_sizes=(3,2), activation="relu", early_stopping=False, max_iter=2000)
classifier_early = MLPClassifier(hidden_layer_sizes=(3,2), activation="relu", early_stopping=True, max_iter=2000)
predictions = []
predictions_early = []
truth = []

for train_i, test_i in kfold.split(breast_X):
    train_data, train_target = breast_X[train_i], breast_targets[train_i]
    test_data, test_target = breast_X[test_i], breast_targets[test_i]
    classifier.fit(train_data, train_target)
    classifier_early.fit(train_data, train_target)
    predictions += list(classifier.predict(test_data))
    predictions_early += list(classifier_early.predict(test_data))
    truth += list(test_target)

conf_m = confusion_matrix(np.array(truth), np.array(predictions), labels=classifier.classes_)
conf_m_early = confusion_matrix(np.array(truth), np.array(predictions_early), labels=classifier_early.classes_)

disp = ConfusionMatrixDisplay(conf_m, display_labels=classifier.classes_)

disp_e = ConfusionMatrixDisplay(conf_m_early, display_labels=classifier_early.classes_)

disp.plot()
plt.title("Classificador Normal")
plt.show()

disp_e.plot()
plt.title("Classificador Early")
plt.show()

#3
kfold = KFold(n_splits=5, random_state=0, shuffle=True)

regressor = MLPRegressor(hidden_layer_sizes=(3,2), activation="relu", alpha=10, max_iter=2000)
regressor_no_reg = MLPRegressor(hidden_layer_sizes=(3,2), activation="relu", alpha=0, max_iter=2000)
residuals = []
residuals_no_reg = []

for train_i, test_i in kfold.split(breast_X):
    train_data, train_target = breast_X[train_i], breast_targets[train_i]
    test_data, test_target = breast_X[test_i], breast_targets[test_i]
    regressor.fit(train_data, train_target)
    regressor_no_reg.fit(train_data, train_target)
    residuals += list(test_target - regressor.predict(test_data))
    residuals_no_reg += list(test_target - regressor_no_reg.predict(test_data))

plt.boxplot(np.array(residuals))
plt.title("Residuals With Regularization")
plt.show()

plt.boxplot(np.array(residuals_no_reg))
plt.title("Residuals Without Regularization")
plt.show()