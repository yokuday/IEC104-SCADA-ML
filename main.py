# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

import pickle

from sklearn.tree import DecisionTreeRegressor

# Load the training and testing datasets from local storage
# You need to specify the path to the files you want to load
# For example, if the files are in the same folder as this script, you can use
# train_path = "Balanced_IEC104_Train_CSV_Files.csv"
# test_path = "Balanced_IEC104_Test_CSV_Files.csv"
# If the files are in a different folder, you need to provide the full path
# For example, if the files are in C:\Users\user\Documents, you can use
# train_path = "C:\\Users\\user\\Documents\\Balanced_IEC104_Train_CSV_Files.csv"
# test_path = "C:\\Users\\user\\Documents\\Balanced_IEC104_Test_CSV_Files.csv"
# You can also use forward slashes instead of backslashes
# For example, train_path = "C:/Users/user/Documents/Balanced_IEC104_Train_CSV_Files.csv"
# Make sure the file names match the ones you have downloaded
train_path = 'some_dir\\Balanced_IEC104_Train_Test_CSV_Files\\iec104_train_test_csvs\\' #"Balanced_IEC104_Train_CSV_Files.csv"
test_path = 'some_dir\\Balanced_IEC104_Train_Test_CSV_Files\\iec104_train_test_csvs\\tests_cic_90\\test_90_cicflow.csv' #"Balanced_IEC104_Test_CSV_Files.csv"

# unaltered packets
#train_files = ['tests_cic_120\\iec104_CICFLOWMETER_train_final.csv', 'tests_cic_90\\test_90_cicflow.csv', 'tests_cic_60\\test_60_CICIFlow.csv', 'tests_cic_30\\test_30_cicflow.csv', 'tests_cic_15\\test_15_cicflow.csv', 'tests_cic_180\\test_180_cicflow.csv'];

# modified packets
train_files = ['tests_custom_15\\test_15_custom_script.csv', 'tests_custom_30\\test_30_custom_script.csv', 'tests_custom_60\\test_60_custom_script.csv', 'tests_custom_90\\test_90_custom_script.csv', 'tests_custom_180\\test_180_custom_script.csv', 'tests_custom_120\\iec104_custom_script_test_final.csv']

dfs = [pd.read_csv(train_path+file) for file in train_files]
dfs2 = [pd.read_csv(train_path+file.replace('train', 'test')) for file in train_files]

#train_df = pd.read_csv(train_path)
train_df = pd.concat(dfs, ignore_index=True)

#test_df = pd.read_csv(test_path)
test_df = pd.concat(dfs2, ignore_index=True)

# Create an instance of LabelEncoder
le = LabelEncoder()


def get_label(df):
    # Get the unique values from the 'labels' column
    labels = df['Label'].unique()

    # Print the labels
    print(labels)


get_label(train_df)

# Fit and transform the "Label" column of train_df
train_df["Label"] = le.fit_transform(train_df["Label"])

# Transform the "Label" column of test_df using the same encoder
test_df["Label"] = le.transform(test_df["Label"])

get_label(train_df)

# Drop the rows with missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Or impute the missing values with 'Normal'
train_df = train_df.fillna('Normal')
test_df = test_df.fillna('Normal')

# Assign column names
# The column names are taken from the [7](https://github.com/topics/intrusion-detection-system?l=python) file[^1^][1]
#train_df.columns = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]
test_df.columns = train_df.columns

# Drop the categorical features ( only needed for the unmodified apcket dataset )
#train_df = train_df.drop(["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"], axis=1)
#test_df = test_df.drop(["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"], axis=1)

# Scale the numerical features
scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df)
test_df_scaled = scaler.transform(test_df)

# Split the data into features and labels
X_train = train_df_scaled[:, :-1]
y_train = train_df_scaled[:, -1]
X_test = test_df_scaled[:, :-1]
y_test = test_df_scaled[:, -1]

# Train the OCSVM model on the training data
#ocsvm = OneClassSVM(kernel="rbf", nu=0.5, gamma=0.0001)
ocsvm = DecisionTreeRegressor(criterion='squared_error', random_state=1, splitter='random', max_depth=50, max_features=5)
#ocsvm = KMeans(1)
#ocsvm = Ridge(alpha=2, fit_intercept=False)

#ocsvm.fit(X_train)
ocsvm.fit(X_train, y_train)

# Predict the labels of the testing data
y_pred = ocsvm.predict(X_test)

y_test = (y_test >= 0).astype(int)
y_pred = (y_pred >= 0).astype(int)

# Evaluate the performance of the model
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
print("Accuracy:", acc)
print("F1-score:", f1)

tpr_0 = recall_score(y_test, y_pred, pos_label=0)
print(tpr_0)

with open('model.pkl', 'wb') as f:
    pickle.dump(ocsvm, f)
