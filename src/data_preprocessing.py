import os
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import RandomOverSampler

# Carica il dataset direttamente dal file CSV
def load_dataset():

    #Percorso del file CSV
    csv_path = "data/StudentPerformanceFactors.csv"

    #Controlla se il file esiste
    if not os.path.exists(csv_path):
         raise FileNotFoundError(f"File {csv_path}non trovato")

    #Carica il dataset in un DataFrame
    print(f"Caricando {csv_path} ...")
    df = pd.read_csv(csv_path)
    print("Dataset caricato ...")
    print(df.head())

    return df

# Converte in categorie numeriche
def encode_categories(df, all=True):
    yes_no_columns = [
        'Extracurricular_Activities',
        'Internet_Access',
        'Learning_Disabilities',

    ]
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"Colonna '{col}' codificata: 'Yes' -> 1, 'No' -> 0")

    df['School_Type'] = df['School_Type'].map({'Private' : 1, 'Public' : 0})
    print("Colonna School_Type codificata: 'Private' -> 1, 'Public' -> 0")

    if all:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        print("Colonna Gender codificata: 'Male' -> 1, 'Female' -> 0")

    low_medium_high_columns = [
        'Parental_Involvement',
        'Access_to_Resources',
        'Motivation_Level',
        'Family_Income',
        'Teacher_Quality'
    ]

    for col in low_medium_high_columns:
        df[col] = df[col].map({'Low': 0, 'Medium': 1, 'High': 2})
        print(f"Colonna '{col}' codificata: 'Low' -> 0, 'Medium' -> 1, 'High' -> 2")

    df['Peer_Influence'] = df['Peer_Influence'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
    print(f"Colonna 'Peer_Influence' codificata: 'Negative' -> 0, 'Neutral' -> 1, 'Positive' -> 2")

    # Mapping per Parental Education Level
    df['Parental_Education_Level'] = df['Parental_Education_Level'].map(
        {'High School': 0, 'College': 1, 'Postgraduate': 2})
    print(f"Colonna 'Parental_Education_Level' codificata: 'High School' -> 0, 'College' -> 1, 'Postgraduate' -> 2")

    # Mapping per Distance from Home
    df['Distance_from_Home'] = df['Distance_from_Home'].map({'Near': 0, 'Moderate': 1, 'Far': 2})
    print(f"Colonna 'Distance_from_Home' codificata: 'Near' -> 0, 'Moderate' -> 1, 'Far' -> 2")

# Identifica gli outlier in una colonna usando l'IQR.
def detect_outliners_iqr(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (df[column] < lower_bound) | (df[column] > upper_bound)

    num_outliers = mask.sum()

    # Stampa il numero di outlier
    print(f"Il numero di outlier nella colonna '{column}' è: {num_outliers}")


def remove_outliers_iqr(df, column):
    for col in column:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        # Mantieni solo i valori entro i limiti
        df = df[mask]
        print(f"Righe dopo la rimozione degli outlier per '{col}': {df.shape[0]}")
    return df

# Raggruppa i valori di una colonna in intervalli specificati.
def group_target_intervals(df, column):
    discretizer = KBinsDiscretizer(n_bins=3,encode='ordinal', strategy='uniform')
    df[column] = discretizer.fit_transform(df[[column]])

    return df

def oversample_to_balance(x,y):
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x,y)

    return x_resampled,y_resampled

