import pandas as pd
import numpy as np
import csv
import os
import time
directory = ' '

# protest, done
protest_df = pd.DataFrame()

# Iterate through files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern
    if filename.startswith("doca_llama33_protest_batch") and filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        # Append the DataFrame to the combined DataFrame
        protest_df = pd.concat([protest_df, df], ignore_index=True)
        print(f"Successfully read: {filename}")
    
protest_df.shape # (2095, 4)
protest_df.to_csv('/protest_llama33.csv')

# police, done
police_df = pd.DataFrame()

# Iterate through files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern
    if filename.startswith("doca_llama33_police") and filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        # Append the DataFrame to the combined DataFrame
        police_df = pd.concat([police_df, df], ignore_index=True)
        print(f"Successfully read: {filename}")

police_df.shape # (2141, 5)
police_df.to_csv('/police_llama33.csv')

# participants
participants_df1 = pd.DataFrame()

# Iterate through files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern
    if filename.startswith("doca_llama33_participants") and filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        # Append the DataFrame to the combined DataFrame
        participants_df1 = pd.concat([participants_df1, df], ignore_index=True)
        print(f"Successfully read: {filename}")

participants_df1.shape # (1645, 4)
participants_df1.to_csv('/participants_llama33_part1.csv')

articles = pd.read_csv("doca_llama3_police_processed.csv")

# Filter rows in articles that do not have a match in participants_df1 based on 'fulltext'
tobedone = articles[~articles['fulltext'].isin(participants_df1['fulltext'])]
tobedone = tobedone[['eventid', 'uid', 'fulltext']]
tobedone.to_csv("participants_tbd.csv", index=False)
print("Rows that do not match have been saved to 'tobedone.csv' with 'eventid', 'uid', and 'fulltext'.")

# Did all the tobedone rows
participants_df = pd.DataFrame()

for filename in os.listdir(directory):
    # Check if the file matches the pattern
    if filename.startswith("doca_llama33_participants") and filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        print(df.shape)
        # Append the DataFrame to the combined DataFrame
        participants_df = pd.concat([participants_df, df], ignore_index=True)
        print(f"Successfully read: {filename}")

participants_df.shape # (2095, 4)
participants_df.to_csv('/participants_llama33.csv')
