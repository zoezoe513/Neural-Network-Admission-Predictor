import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Binary target
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
    
    # Drop Serial_No
    if 'Serial_No' in df.columns:
        df = df.drop('Serial_No', axis=1)

    # Encode categorical features
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype=int)

    X = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

    # Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
