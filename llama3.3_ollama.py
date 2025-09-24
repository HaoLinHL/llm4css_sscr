'''
$ ssh haolin1@login.seawulf.stonybrook.edu
$ ssh -X haolin1@milan
$ ssh -X haolin1@dn-rome1

$ ssh haolin1@dg-mem.seawulf.stonybrook.edu

$ module load ollama/0.1.44-amd
$ module load ollama-py/0.3.3
$ ollama list
$ ollama pull llama3.3
$ ollama run llama3.3

$ module load ollama/0.1.44-amd
$ module load anaconda
$ conda activate llama
$ pip install ollama
$ python

$ tmux a -t 0
[detached (from session 1)]
'''

import requests
import json
import pandas as pd
import csv
import os
import time
import ollama

# Function to send prompt to LLaMA model
def llama3(prompt):
    response = ollama.chat(
        model='llama3.3',
        messages=[{
            'role': 'user',
            'system':'You are an expert trained to extract protest-related information from news articles.',
            'content': prompt
        }]
    )
    return response['message']['content']

# Load the dataset
# df = pd.read_csv("/Users/lin/Documents/projects/llm_classification/with_josh_social_movement/llama/doca_llama3_police_processed.csv")
df = pd.read_csv("/gpfs/scratch/haolin1/doca_llama3_police_processed.csv")
print("DataFrame shape:", df.shape)

# Define output paths and prompts
tasks = [
    {
        "output_div": "/gpfs/scratch/haolin1/doca_llama33_police.csv",
        "prompt": "Analyze the following article and determine if it mentions the presence of police at the event. Respond in the format: `< 1 >` if police are mentioned or `< 0 >` if they are not. Let us think step by step. Here is the news article:"
    },
    {
        "output_div": "/gpfs/scratch/haolin1/doca_llama33_participants.csv",
        "prompt": "Analyze the following article to determine the number of participants at the event. Categorize the number of participants using the following scale: 1. Small, handful (1–9 people) 2. Group, committee (10–49 people) 3. Large gathering (50–99 people) 4. Hundreds, mass, mob (100–999 people) 5. Thousands (1,000–9,999 people) 6. Tens of thousands (10,000 or more people). Respond with the numeric category (1–6) in the format: `< numeric answer >`. Let us think step by step. Here is the news article:"
    },
    {
        "output_div": "/gpfs/scratch/haolin1/dprotest activitiesoca_llama33_protest.csv",
        "prompt": "Analyze the  described in the following article. Based on the details, classify the activities into one or more of the categories below. Use the numeric codes provided to represent the activities, and you may assign up to four categories. Categories of Protest Activities: 1. Bannering (e.g., hanging banners on buildings, bridges) 2. Bell ringing 3. Bicycling in a protest procession 4. Candle-lighting or carrying/displaying candles 5. Canvassing (e.g., asking for votes, signatures, money, opinions) 6. Cross carrying 7. Dancing in celebration for peace 8. Debate 9. Public and collective discussion 10. Dramaturgical presentation (e.g., skits, street theater, puppets) 11. Fasting or hunger strikes 12. Film showing 13. Fireworks display 15. Leafleting (e.g., distributing literature) 16. Silent meditation or service 17. Chariot parading 18. Petitioning (e.g., obtaining or presenting signatures) 19. Photo exhibiting (e.g., images of atrocities/victims) 20. Holding signs, picketing, placarding 21. Praying 22. Procession or marching 23. Reading/reciting documents (e.g., Declaration of Independence) 25. Selling protest items (e.g., buttons, books) 26. Silence (e.g., silent vigil) 27. Speechmaking (e.g., talks, testimonies) 28. Sloganeering/chanting 29. Vigiling (e.g., silent protests with placards/banners) 31. Worship-like protest services 32. Wreath-laying or floral offerings 33. Symbolic or artistic displays 34. Press conferences 35. Ceremonial activities 36. Musical or vocal performances 37. Filming/photographing events 38. Recruiting/evangelizing for movements 39. Camping/erecting tents overnight 40. Lobbying local/state/federal governments 41. Polling opinions on social movements 42. Singing collectively 43. Torch or item passing (e.g., relay style) 44. Bed racing 45. Civil disobedience (illegal acts as protest) 46. Meeting political candidates 48. Flag waving 49. Distributing goods 50. Describing projects 51. Drumming 52. Sit-ins 53. Economic protests (e.g., bank-ins, shop-ins) 54. Withholding obligations (e.g., work, rent, taxes) 55. Physical attacks 56. Verbal attacks or threats 57. Blockades 58. Loud noise-making 59. Yelling/shouting 60. Building takeovers 61. Looting 62. Property damage 63. Kidnapping or hostage-taking 64. Meeting disruptions 65. Walk-outs (e.g., from meetings, ceremonies) 66. Letter-writing campaigns 97. Legal maneuvers (e.g., lawsuits) 98. Other activities 99. Missing activity Instructions: 1. Identify up to four categories of protest activities present in the article. 2. Respond in the format `< numeric answer 1 ; numeric answer 2 ; numeric answer 3 ; numeric answer 4 >`. 3. If fewer than four categories apply, leave the remaining slots empty (e.g., `< 1 ; 20 ; ; >`). Let us think step by step. Here is the news article:"
    }
]


results = []

for task in tasks:
    start = time.time()
    print(f"Starting task for {task['output_div']}...")
    results = []  # Collect rows in memory for this task
    batch_count = 0  # Counter to track batches
    try:
        for index, row in df.iterrows():
            eventid = row['eventid']
            fulltext = row['fulltext']
            uid = row['uid']
            prompt = f"{task['prompt']} {fulltext}"
            try:
                # Get output from Llama 3.3
                output = llama3(prompt)
                if not output or output == "Error":
                    print(f"Empty or error output for eventid: {eventid}, uid: {uid}")
                    output = "Error"
                # Append the result to the list
                results.append({
                    'eventid': eventid,
                    'fulltext': fulltext,
                    'uid': uid,
                    'output': output
                })
                # Increment batch count and write to CSV if batch size is 50
                if len(results) == 50:
                    batch_count += 1
                    batch_filename = f"{task['output_div'].replace('.csv', '')}_batch_{batch_count}.csv"
                    with open(batch_filename, 'w', newline='') as csvfile:
                        fieldnames = ['eventid', 'fulltext', 'uid', 'output']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"Batch {batch_count} written to {batch_filename}")
                    # Clear the results for the next batch
                    results = []
                print(f"Processed row {index + 1}/{len(df)}")
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                results.append({
                    'eventid': eventid,
                    'fulltext': fulltext,
                    'uid': uid,
                    'output': "Error"
                })
        # Write any remaining results after the last batch
        if results:
            batch_count += 1
            batch_filename = f"{task['output_div'].replace('.csv', '')}_batch_{batch_count}.csv"
            with open(batch_filename, 'w', newline='') as csvfile:
                fieldnames = ['eventid', 'fulltext', 'uid', 'output']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Final batch written to {batch_filename}")
    except Exception as e:
        print(f"Error during task for {task['output_div']}: {e}")
    end = time.time()
    print(f"Task completed for {task['output_div']} in {end - start:.2f} seconds.")

# 6967.67 seconds for df[1650:]
# police

# Task completed for /gpfs/scratch/haolin1/doca_llama33_protest.csv in 31383.27 seconds.
    
# Processed row 450/450
# Task completed for /gpfs/scratch/haolin1/doca_llama33_participants.csv in 6420.22 seconds.
# Total number of row 2088