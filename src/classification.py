import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Visualizza le metriche di valutazione per ogni modello
def visualizeMetricsGraphs(model_results):
    models = list(model_results.keys())

    accuracy = np.array([model_results[clf]['Accuracy'] for clf in models])
    precision = np.array([model_results[clf]['Precision (macro)'] for clf in models])
    recall = np.array([model_results[clf]['Recall (macro)'] for clf in models])
    f1 = np.array([model_results[clf]['F1 Score (macro)'] for clf in models])

    bar_width = 0.2
    index = np.arange(len(models))

    plt.figure(figsize=(10, 6))
    plt.bar(index, accuracy, bar_width, label='Accuracy', color='#4a90e2')
    plt.bar(index + bar_width, precision, bar_width, label='Precision', color='#50e3c2')
    plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='#7b4397')
    plt.bar(index + 3 * bar_width, f1, bar_width, label='F1', color='#0f4c81')

    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title('Evaluation Metrics for Each Model', fontsize=14)
    plt.xticks(index + 1.5 * bar_width, models)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Divide il dataset in train e test sets
def split_dataset(x, y, test_size=0.2, random_state=42):

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def optimize_decision_tree(x, y):
    decision_tree_parameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [5, 10, 15],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20,50],
        'splitter': ['best'],
    }
    """"{
'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'splitter': ['best'],

    }"""
    grid_search_dc = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        decision_tree_parameters,
        cv=5,
        scoring='f1_macro',
    )
    grid_search_dc.fit(x, y)
    return grid_search_dc.best_estimator_, grid_search_dc.best_params_


def optimize_random_forest(x, y):
    random_forest_parameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [5, 10, 15],
        'n_estimators': [50, 100, 150],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20, 50],
    }
    """"{'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 5, 10],
        'n_estimators': [10, 20, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20],
                
    }"""
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        random_forest_parameters,
        cv=5,
        scoring='f1_macro',
    )
    grid_search_rf.fit(x, y)
    return grid_search_rf.best_estimator_, grid_search_rf.best_params_


def optimize_logistic_regression(x, y):
    logistic_regression_parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100000, 150000],
    }
    grid_search_lr = GridSearchCV(
        LogisticRegression(random_state=42),
        logistic_regression_parameters,
        cv=5,
        scoring='f1_macro',
    )
    grid_search_lr.fit(x, y)
    return grid_search_lr.best_estimator_, grid_search_lr.best_params_


# Valuta un modello sul test set e stampa il report di classificazione insieme alla metrica di scoring selezionata.
def evaluate_model(model, x_test, y_test, scoring_metric):
    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if scoring_metric == "accuracy":
        score = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {score:.2f}")
    elif scoring_metric == "precision_macro":
        score = precision_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"Precision (macro): {score:.2f}")
    elif scoring_metric == "recall_macro":
        score = recall_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"Recall (macro): {score:.2f}")
    elif scoring_metric == "f1_macro":
        score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"F1 Score (macro): {score:.2f}")
    else:
        print("Unknown scoring metric.")
        score = None
    return score


def generate_model_parameters_table(dt_params, rf_params, lr_params):
    data = {
        "Modello": [
            "Decision Tree", "Decision Tree", "Decision Tree", "Decision Tree",
            "Random Forest", "Random Forest", "Random Forest", "Random Forest", "Random Forest",
            "Logistic Regression", "Logistic Regression", "Logistic Regression", "Logistic Regression"
        ],
        "Parametro": [
            "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
            "criterion", "max_depth", "n_estimators", "min_samples_split", "min_samples_leaf",
            "C", "penalty", "solver", "max_iter"
        ],
        "Valore": [
            dt_params['criterion'], dt_params['max_depth'], dt_params['min_samples_split'],
            dt_params['min_samples_leaf'],
            rf_params['criterion'], rf_params['max_depth'], rf_params['n_estimators'], rf_params['min_samples_split'],
            rf_params['min_samples_leaf'],
            lr_params['C'], lr_params['penalty'], lr_params['solver'], lr_params['max_iter']
        ]
    }
    df = pd.DataFrame(data)
    print(df)

def generate_model_parameters_table2(dt_params, rf_params):
    data = {
        "Modello": [
            "Decision Tree", "Decision Tree", "Decision Tree", "Decision Tree",
            "Random Forest", "Random Forest", "Random Forest", "Random Forest", "Random Forest"
        ],
        "Parametro": [
            "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
            "criterion", "max_depth", "n_estimators", "min_samples_split", "min_samples_leaf"
        ],
        "Valore": [
            dt_params['criterion'], dt_params['max_depth'], dt_params['min_samples_split'],
            dt_params['min_samples_leaf'],
            rf_params['criterion'], rf_params['max_depth'], rf_params['n_estimators'], rf_params['min_samples_split'],
            rf_params['min_samples_leaf']
        ]
    }
    df = pd.DataFrame(data)
    print(df)

# Trova i migliori iperparametri con k-fold CV nel training set e valuta i modelli sul test set

def train_models(x_train, y_train, x_test, y_test):
    model_results = {
        'DecisionTree': {},
        'RandomForest': {},
        'LogisticRegression': {}
    }

    dt, p_dt = optimize_decision_tree(x_train, y_train)
    rf, p_rf = optimize_random_forest(x_train, y_train)


    scaler = StandardScaler()
    cols_to_standardize = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                           'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']

    x_train_lr = x_train.copy()
    x_train_lr[cols_to_standardize] = scaler.fit_transform(x_train_lr[cols_to_standardize])

    x_test_lr = x_test.copy()
    x_test_lr[cols_to_standardize] = scaler.transform(x_test_lr[cols_to_standardize])

    lr, p_lr = optimize_logistic_regression(x_train_lr, y_train)

    generate_model_parameters_table(p_dt, p_rf, p_lr)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results_dt = {}
    results_rf = {}
    results_lr = {}

    for scoring_metric in scoring_metrics:
        print(f"\n--- Evaluating {scoring_metric} ---")
        dt_score = evaluate_model(dt, x_test, y_test, scoring_metric)
        rf_score = evaluate_model(rf, x_test, y_test, scoring_metric)
        lr_score = evaluate_model(lr, x_test_lr, y_test, scoring_metric)

        results_dt[scoring_metric] = dt_score
        results_rf[scoring_metric] = rf_score
        results_lr[scoring_metric] = lr_score

    model_results['DecisionTree'] = {
        'Accuracy': results_dt['accuracy'],
        'Precision (macro)': results_dt['precision_macro'],
        'Recall (macro)': results_dt['recall_macro'],
        'F1 Score (macro)': results_dt['f1_macro']
    }
    model_results['RandomForest'] = {
        'Accuracy': results_rf['accuracy'],
        'Precision (macro)': results_rf['precision_macro'],
        'Recall (macro)': results_rf['recall_macro'],
        'F1 Score (macro)': results_rf['f1_macro']
    }
    model_results['LogisticRegression'] = {
        'Accuracy': results_lr['accuracy'],
        'Precision (macro)': results_lr['precision_macro'],
        'Recall (macro)': results_lr['recall_macro'],
        'F1 Score (macro)': results_lr['f1_macro']
    }

    results_table = pd.DataFrame({
        'Model': ['DecisionTree', 'RandomForest', 'LogisticRegression'],
        'Accuracy': [
            model_results['DecisionTree']['Accuracy'],
            model_results['RandomForest']['Accuracy'],
            model_results['LogisticRegression']['Accuracy']
        ],
        'Precision (macro)': [
            model_results['DecisionTree']['Precision (macro)'],
            model_results['RandomForest']['Precision (macro)'],
            model_results['LogisticRegression']['Precision (macro)']
        ],
        'Recall (macro)': [
            model_results['DecisionTree']['Recall (macro)'],
            model_results['RandomForest']['Recall (macro)'],
            model_results['LogisticRegression']['Recall (macro)']
        ],
        'F1 Score (macro)': [
            model_results['DecisionTree']['F1 Score (macro)'],
            model_results['RandomForest']['F1 Score (macro)'],
            model_results['LogisticRegression']['F1 Score (macro)']
        ]
    })
    print("\nEvaluation Metrics Table:")
    print(results_table)

    visualizeMetricsGraphs(model_results)
    print('Model training and evaluation complete.')
    return model_results


# Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, x, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=10, scoring='f1_macro')

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(
        f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='#4a90e2')  # Blu
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='#FF0000')   # Viola
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()


def train_model_kfold(x,y):

    model = {
        'DecisionTree': {
            'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1_list': [],
        },

        #'RandomForest': {
         #   'accuracy_list': [],
           # 'precision_list': [],
          #  'recall_list': [],
            #'f1_list': [],
        #}
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    dt, p_dt = optimize_decision_tree(x, y)
    #rf, p_rf = optimize_random_forest(x, y)

   # generate_model_parameters_table2(p_dt,p_rf)

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results_dt = {}
    results_rf = {}

    for scoring_metric in scoring_metrics:
        scoring_metric_dt = cross_val_score(dt, x, y, cv=cv, scoring=scoring_metric)
        #scoring_metric_rf = cross_val_score(rf, x, y, cv=cv, scoring=scoring_metric)

        #results_rf[scoring_metric] = scoring_metric_rf
        results_dt[scoring_metric] = scoring_metric_dt

    model['DecisionTree']['accuracy_list'] = (results_dt['accuracy'])
    model['DecisionTree']['precision_list'] = (results_dt['precision_macro'])
    model['DecisionTree']['recall_list'] = (results_dt['recall_macro'])
    model['DecisionTree']['f1_list'] = (results_dt['f1_macro'])
   #model['RandomForest']['accuracy_list'] = (results_rf['accuracy'])
    #model['RandomForest']['precision_list'] = (results_rf['precision_macro'])
    #model['RandomForest']['recall_list'] = (results_rf['recall_macro'])
    #model['RandomForest']['f1_list'] = (results_rf['f1_macro'])



    plot_learning_curves(dt, x, y, 'DecisionTree')
   # plot_learning_curves(rf, x, y, 'RandomForest')
    visualizeMetricsGraphs_Kfold(model)

    print('Allenamento fatto')
    print(model)
    return model


# Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs_Kfold(model):
    models = list(model.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([model[clf]['accuracy_list'] for clf in models])
    precision = np.array([model[clf]['precision_list'] for clf in models])
    recall = np.array([model[clf]['recall_list'] for clf in models])
    f1 = np.array([model[clf]['f1_list'] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_precision = np.mean(precision, axis=1)
    mean_recall = np.mean(recall, axis=1)
    mean_f1 = np.mean(f1, axis=1)

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    plt.bar(index, mean_accuracy, bar_width, label='Accuracy', color='#4a90e2')   # Blu
    plt.bar(index + bar_width, mean_precision, bar_width, label='Precision', color='#50e3c2')  # Celeste
    plt.bar(index + 2 * bar_width, mean_recall, bar_width, label='Recall', color='#7b4397')    # Viola
    plt.bar(index + 3 * bar_width, mean_f1, bar_width, label='F1', color='#0f4c81')            # Blu scuro

    # Aggiunta di etichette e legenda
    plt.xlabel('Modelli')
    plt.ylabel('Punteggi medi')
    plt.title('Punteggio medio per ogni modello')
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    # Visualizzazione del grafico
    plt.show()