import pandas as pd

url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$where=year>=2022&$limit=600000"
df = pd.read_csv(url)

print(df.shape)
print(df['year'].value_counts())

# Save locally, work from this file from now on
df.to_csv('chicago_crimes_2022_2026.csv', index=False)