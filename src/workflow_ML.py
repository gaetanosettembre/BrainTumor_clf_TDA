# -*- coding: utf-8 -*-
"""

@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils import import_data, data_to_negative, display_image_grid
from utils import plot_confusion_matrix, fit_predict_classifier, evaluate_classifier
from utils import compute_tucker_decomposition
from utils import plot_rank_size

#from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import tensorly as tl
import time

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Set Training and Test set path
path_train = 'Dataset/Training'
path_test = 'Dataset/Testing'

# Import Training and Test Set
x_train, y_train = import_data(path_train, labels, 250)
x_test, y_test = import_data(path_test, labels, 250)

# Make images negative
x_train = data_to_negative(x_train)
x_test = data_to_negative(x_test)

# Normalization
x_train = np.array(x_train) / 255.0  # normalize Images into range 0 to 1.
x_test = np.array(x_test) / 255.0

# Show some images in Training Dataset
images = [x_train[i] for i in range(36)]
images_to_display = [x_train[i] for i in range(36)]
display_image_grid(images_to_display, grid_size=(6, 6), figsize=(10, 10), cmap="gray", title="Image Grid of data")


# Dataset in matrix form
X_train = np.vstack([image.flatten() for image in x_train])
X_test = np.vstack([image.flatten() for image in x_test])


# Definizione dei modelli
model_constructors = {
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "Random Forest": RandomForestClassifier,
    "Extra Trees": ExtraTreesClassifier,
    "XGBoost": XGBClassifier,
    "AdaBoost": AdaBoostClassifier
}


# # Inizializza un dizionario per salvare i risultati di tutte le iterazioni
# all_results = {model_name: {"Accuracy": [], "F1-Score": [], "Precision": [], "Recall": [], "Training Time": []} for model_name in models.keys()}

# for _ in range(num_iteration):
#     results = {}
#     for name, model in models.items():
#         start_time = time.time()
#         model.fit(X_train, y_train)
#         train_time = time.time() - start_time
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         precision = precision_score(y_test, y_pred, average='weighted')
#         recall = recall_score(y_test, y_pred, average='weighted')
#         results[name] = {
#             "Accuracy": accuracy,
#             "F1-Score": f1,
#             "Precision": precision,
#             "Recall": recall,
#             "Training Time": train_time
#         }
            
#         # Aggiungi i risultati di questa iterazione al dizionario complessivo
#         for metric, value in results[name].items():
#             all_results[name][metric].append(value)

# average_results = {}
# std_dev_results = {}
# for name, metrics in all_results.items():
#     average_results[name] = {metric: np.mean(values) for metric, values in metrics.items()}
#     std_dev_results[name] = {metric: np.std(values) for metric, values in metrics.items()}

# # Stampa le medie e le deviazioni standard
# for name, metrics in average_results.items():
#     print(f"Modello: {name}")
#     print(f"Media Accuracy: {metrics['Accuracy']:.4f} (Deviazione Standard: {std_dev_results[name]['Accuracy']:.4f})")
#     print(f"Media F1-Score: {metrics['F1-Score']:.4f} (Deviazione Standard: {std_dev_results[name]['F1-Score']:.4f})")
#     print(f"Media Precision: {metrics['Precision']:.4f} (Deviazione Standard: {std_dev_results[name]['Precision']:.4f})")
#     print(f"Media Recall: {metrics['Recall']:.4f} (Deviazione Standard: {std_dev_results[name]['Recall']:.4f})")
#     print(f"Media Tempo di addestramento: {metrics['Training Time']:.2f} secondi")
#     print("---------------------")



# Definisci le metriche da calcolare
metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score
}

# Addestra e valuta i modelli
for name, model_constructor in model_constructors.items():
    print(f"Training {name}...")
    
    scores = {metric_name: [] for metric_name in metrics}
    train_times = []

    for _ in range(5):  # Esegui 5 volte per avere una media delle prestazioni
        model = model_constructor()    
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        
        train_times.append(training_time)
        
        for metric_name, metric_func in metrics.items():
            if metric_name == "Accuracy":
                score = metric_func(y_test, y_pred)
            else:
                score = metric_func(y_test, y_pred, average='weighted')
            scores[metric_name].append(score)
    
    # Calcola le medie delle prestazioni
    avg_scores = {metric_name: np.mean(score_list) for metric_name, score_list in scores.items()}
    std_scores = {metric_name: np.std(score_list) for metric_name, score_list in scores.items()}
    avg_times = np.mean(train_times)
    std_times = np.std(train_times)
    
    # Stampa le prestazioni medie e deviazione standard
    print(f"Average Performance of {name}:")
    for metric_name, avg_score in avg_scores.items():
        std_score = std_scores[metric_name]
        print(f"{metric_name}: {avg_score:.4f} ± {std_score:.4f}")
    print(f"Training time stats: {avg_times:.4f} ± {std_times:.4f}")
    
    print(f"{name} finished training.\n")
    
    
    

x_train_fold = tl.fold(X_train, mode=2, shape=(250, 250, x_train.shape[0]))
x_test_fold = tl.fold(X_test, mode=2, shape=(250, 250, x_test.shape[0]))


# Compute Tucker Decomposition
# !!Attention!!: Use (10, 10, 300) if you want train and use RF. rank_decomposition = (30, 30, 300) for SVM
rank_decomposition = (30, 30, 300)
core, U0, U1, U2 = compute_tucker_decomposition(rank_decomposition, x_train_fold)

# Projected core tensor on images basis
pU2 = core@U2.T
core_p = tl.unfold(pU2, mode=2)

# Build test features using first two decomposition basis
step1 = tl.tenalg.mode_dot(x_test_fold, np.linalg.pinv(U0), mode=0)
step2 = tl.tenalg.mode_dot(step1, np.linalg.pinv(U1), mode=1)
x_test_reprojected = step2
x_test_unfolded = tl.unfold(x_test_reprojected, mode=2)

# Addestra e valuta i modelli
for name, model_constructor in model_constructors.items():
    print(f"Training {name}...")
    
    scores = {metric_name: [] for metric_name in metrics}
    train_times = []

    for _ in range(5):  # Esegui 5 volte per avere una media delle prestazioni
        model = model_constructor()    
        start_time = time.time()
        model.fit(core_p, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(x_test_unfolded)
        
        train_times.append(training_time)
        
        for metric_name, metric_func in metrics.items():
            if metric_name == "Accuracy":
                score = metric_func(y_test, y_pred)
            else:
                score = metric_func(y_test, y_pred, average='weighted')
            scores[metric_name].append(score)
    
    # Calcola le medie delle prestazioni
    avg_scores = {metric_name: np.mean(score_list) for metric_name, score_list in scores.items()}
    std_scores = {metric_name: np.std(score_list) for metric_name, score_list in scores.items()}
    avg_times = np.mean(train_times)
    std_times = np.std(train_times)
    
    # Stampa le prestazioni medie e deviazione standard
    print(f"Average Performance of {name}:")
    for metric_name, avg_score in avg_scores.items():
        std_score = std_scores[metric_name]
        print(f"{metric_name}: {avg_score:.4f} ± {std_score:.4f}")
    print(f"Training time stats: {avg_times:.4f} ± {std_times:.4f}")
    
    print(f"{name} finished training.\n")