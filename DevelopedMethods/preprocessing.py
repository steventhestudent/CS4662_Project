# Data Exploration.ipynb —— all cells pasted, minus graphs & logging   --so just data preprocessing lines
from DevelopedMethods.DF import * # import np, pd, sklearn fn's --and df (https://www.kaggle.com/datasets/kartik2112/fraud-detection/)

# make exploration easier, drop redundant columns:

# manually encode binary/simple labels
# categorical features:

# split trans_date_trans_time into y-m-d-h-m-s columns

# one-hot encoding features w/ few unique values

# Label Encoding (convert value into unique int) - good for ft. w/ many unique values
# label_enc = LabelEncoder()
# df['merchant'] = label_enc.fit_transform(df['merchant']) # 693 unique values


# smote = SMOTE(sampling_strategy=0.2, random_state=42)  # Resample minority class to 20% of majority class
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# X_test_balanced, y_test_balanced = smote.fit_resample(X_test, y_test)

""" this ran before # Feature Matrix (X) & Target/label Vector (y), but we cannot SMOTE (imbalance handling) non-discrete/continuous values (ie: scaling converts nearly everything into a decimal)
 so, we now have to apply to: X_train, X_test (opposed to df_train & df_test) """
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_balanced)
# X_test_scaled = scaler.transform(X_test_balanced)
# Re-wrap the scaled data into a DataFrame with original column names and indices (scaler transforms into numpy.ndarray (NOT a dataframe))
# X_train_balanced = pd.DataFrame(X_train_scaled, columns=X_train_balanced.columns, index=X_train_balanced.index)
# X_test_balanced = pd.DataFrame(X_test_scaled, columns=X_test_balanced.columns, index=X_test_balanced.index)
