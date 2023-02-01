import pandas as pd
from pipeline import TextPreprocessing as MyTextPreprocessing


df = pd.read_csv('dataset\\datasets-labeled-raw.csv')

print(df.head)

# Clean the text
df.dropna(inplace=True,)

df['sentence'] = df['sentence'].apply(lambda x: MyTextPreprocessing(x).clean_text())

print(df.head)

# Write the cleaned data to a new file
df.to_csv('dataset\\datasets-labeled-cleaned.csv', index=False)


