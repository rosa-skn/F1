import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv/results.csv")
df=df.sample(n=100, random_state=1)
print(df.head())
sns.boxplot(data=df, x="raceId", y="milliseconds")
plt.title("Distribution of Race Rounds per Year")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()