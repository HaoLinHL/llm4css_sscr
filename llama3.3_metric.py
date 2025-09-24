
'''
Caculate the Metrics for Llama3.3
'''
import json
import pandas as pd
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# clean data
protest_df = pd.read_csv("/llama/protest_llama33.csv")
police_df = pd.read_csv("/llama/police_llama33.csv")
participants_df = pd.read_csv("/llama/participants_llama33.csv")


police_df['output_clean'] = police_df['output'].str.replace(r'[<>]', '', regex=True) # clean
nan_police = police_df['output_clean'].isna().sum() # 
police_df.to_csv("/llama/police_llama33.csv")

participants_df['output_clean'] = pd.to_numeric(participants_df['output'], errors = 'coerce') # < >
nan_participants = participants_df['output_clean'].isna().sum()
participants_df['output_clean'] = participants_df['output_clean'].fillna(participants_df['output'].str.extract(r'<([^>]*)>')[0])
participants_df['output_clean'] = participants_df['output_clean'].fillna(participants_df['output'].str.extract(r'`([^>]*)`')[0])
participants_df.to_csv("/llama/participants_llama33.csv")

protest_df['output_clean'] = protest_df['output'].str.extract(r'<([^<>]*;[^<>]*)>')
nan_protest = protest_df['output_clean'].isna().sum() # multiple
protest_df.to_csv("/llama/protest_llama33.csv")

# Caculate the Metrix
baseline_div = '/unique_merged_text.csv'
police_div = '/llama/police_llama33.csv'
participant_div = '/llama/participants_llama33.csv'
protest_div = '/llama/protest_llama33.csv'

baseline = pd.read_csv(baseline_div, on_bad_lines='skip')
police = pd.read_csv(police_div)
participant = pd.read_csv(participant_div)
protest = pd.read_csv(protest_div)

# Binary
# presence of police
# variable: police1 
police.rename(columns={'fulltext': 'text'}, inplace=True)
merged_data1 = pd.merge(police, baseline[['eventid', 'text', 'police1']], on=['eventid', 'text'], how='inner')
merged_data1 = pd.merge(police, baseline[['eventid', 'text', 'police1']], on=['eventid'], how='inner')
baseline['text'].is_unique # false
baseline['eventid'].is_unique # false
merged_data1['police1'] = merged_data1['police1'].astype(int)
merged_data1['output_clean'] = merged_data1['output_clean'].astype(int)
# Calculate metrics
f1 = f1_score(merged_data1['police1'], merged_data1['output_clean'])
precision = precision_score(merged_data1['police1'], merged_data1['output_clean'])
recall = recall_score(merged_data1['police1'], merged_data1['output_clean'])
accuracy = accuracy_score(merged_data1['police1'], merged_data1['output_clean'])
conf_matrix = confusion_matrix(merged_data1['police1'], merged_data1['output_clean'])

print(f"Accuracy: {accuracy}") # 0.77
print(f"Precision: {precision}") # 0.44
print(f"Recall: {recall}") # 0.81
print(f"F1 Score: {f1}") # 0.57
print("Confusion Matrix:")
print(conf_matrix)
# [[1292  406]
# [  74  316]]

# Multiclass 
# number of participants
# variable: partices
def categorize_particex(value):
    if 1 <= value <= 9:
        return 1  # Small, handful (1–9 people)
    elif 10 <= value <= 49:
        return 2  # Group, committee (10–49 people)
    elif 50 <= value <= 99:
        return 3  # Large gathering (50–99 people)
    elif 100 <= value <= 999:
        return 4  # Hundreds, mass, mob (100–999 people)
    elif 1000 <= value <= 9999:
        return 5  # Thousands (1,000–9,999 people)
    elif value >= 10000:
        return 6  # Tens of thousands (10,000 or more people)
    else:
        return np.nan  # Handle unexpected values

# Apply the categorization to 'particex' and assign to 'partices' where 'partices' is missing
baseline['partices'] = baseline.apply(
    lambda row: categorize_particex(row['particex']) if pd.isnull(row['partices']) else row['partices'],
    axis=1
)

merged_data2 = pd.merge(participant, baseline[['eventid', 'text', 'partices']], on=['eventid'], how='inner')
merged_data2['partices'] = merged_data2['partices'].astype(int)
merged_data2['output_clean'] = merged_data2['output_clean'].astype(int)
merged_data2['output_clean'] = merged_data2['output_clean'].fillna(99999)
merged_data2['partices'] = merged_data2['partices'].fillna(99999)

print(merged_data2['partices'].value_counts())
print(merged_data2['output_clean'].value_counts())
# Calculate metrics
f1 = f1_score(merged_data2['partices'], merged_data2['output_clean'], average='weighted')
precision = precision_score(merged_data2['partices'], merged_data2['output_clean'], average='weighted')
recall = recall_score(merged_data2['partices'], merged_data2['output_clean'], average='weighted')
accuracy = accuracy_score(merged_data2['partices'], merged_data2['output_clean'])
conf_matrix = confusion_matrix(merged_data2['partices'], merged_data2['output_clean'])
print("Weighted F1 Score:", f1) #  0.43
print("Precision:", precision) # 0.47
print("Recall:", recall) # 0.44
print("Accuracy:", accuracy) # 0.44
print("Confusion Matrix:\n", conf_matrix)

# Multilabel
# protest activities
# variable: act1-act4
merged_data3 = pd.merge(protest, baseline[['eventid', 'text', 'act1','act2','act3','act4']], on=['eventid'], how='inner')

merged_data3['predicted'] = merged_data3['output_clean'].apply(
    lambda x: [int(i) for i in str(x).split(';') if i.strip() and i.strip().lower() != 'nan']
)

merged_data3['true_labels'] = merged_data3[['act1', 'act2', 'act3', 'act4']].apply(
    lambda row: [int(x) for x in row.dropna()], axis=1
)

print(merged_data3[['true_labels', 'predicted']])

# A prediction is considered correct only if all the true labels 
# in true_labels are present in the predicted labels (output_clean).
def is_correct(true_labels, predicted_labels):
    return set(true_labels).issubset(set(predicted_labels))

merged_data3['is_correct'] = merged_data3.apply(
    lambda row: is_correct(row['true_labels'], row['predicted']), axis=1
)

print(merged_data3[['true_labels', 'predicted', 'is_correct']])

merged_data3['is_correct'].value_counts()
# False    1519
# True      568

f1 = f1_score(merged_data3['is_correct'], [1] * len(merged_data3))  # Compare with all correct predictions
print("F1 Score:", f1) # 0.42

accuracy = merged_data3['is_correct'].mean()
print("Accuracy:", accuracy)
# Accuracy: 0.2721609966459032