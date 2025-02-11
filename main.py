from src.bayesian_analysis import *
from src.classification import *
from src.data_preprocessing import *

def main():

    #Carica il dataset
    print("Avvio del caricamento del dataset...")
    data = load_dataset()
    print("Dataset caricato ...")

    #Data procession standard
    data_cleaned = data.copy().dropna()
    data_cleaned = remove_outliers_iqr(data_cleaned,['Hours_Studied', 'Tutoring_Sessions', 'Exam_Score'])

    #Encode
    encode_categories(data_cleaned)

    #Riduciamo le classi in intervalli per Exam Score
    print("\nExam score trasformato a intevalli")
    data_cleaned = group_target_intervals(data_cleaned, 'Exam_Score')

    # Dividi il dataset in feature e target
    target_column = 'Exam_Score'
    x = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    #print("\nDistribuzione delle classi di Exam_Score nel train set:")
    #print(y_train.value_counts().sort_index())

    x_resampled, y_resampled = oversample_to_balance(x_train,y_train)

    #print("\nDistribuzione delle classi di Exam_Score nel dataset:")
    print(y_resampled.value_counts().sort_index())

    model = train_models(x_resampled, y_resampled,x_test, y_test)

    #Bayesian Network

    # Preparaione dati, riduciamo le classi numeriche in intervalli
    data_cleaned_b = group_target_intervals(data_cleaned, 'Hours_Studied')

    data_cleaned_b = group_target_intervals(data_cleaned_b, 'Attendance')

    data_cleaned_b = group_target_intervals(data_cleaned_b, 'Previous_Scores')

    data_cleaned_b = group_target_intervals(data_cleaned_b, 'Tutoring_Sessions')

    data_cleaned_b = group_target_intervals(data_cleaned_b, 'Physical_Activity')

    data_cleaned_b = group_target_intervals(data_cleaned_b, 'Sleep_Hours')

    bayesianNetwork = bayesianNT(data_cleaned_b)
    #bayesianNetwork = load_model()
    visualizeBayesianNetwork(bayesianNetwork)
    visualizeInfo(bayesianNetwork)

    # Generazione di un esempio randomico e predizione della sua classe
    esempioRandom = generate_example(bayesianNetwork)
    print("ESEMPIO RANDOMICO GENERATO --->  ", esempioRandom)
    print("PREDIZIONE DEL SAMPLE RANDOM")
    bayesian_prediction(bayesianNetwork, esempioRandom.to_dict('records')[0], "Exam_Score")

if __name__ == "__main__":
    main()






