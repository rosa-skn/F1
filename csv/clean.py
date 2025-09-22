import pandas as pd

results = pd.read_csv("csv/results.csv")
sprint = pd.read_csv("csv/sprint_results.csv")
drivers = pd.read_csv("csv/driver_results.csv")
pitstops = pd.read_csv("csv/pit_stops.csv")
quali = pd.read_csv("csv/qualifying.csv")
standings = pd.read_csv("csv/constructor_results.csv")

results.head()
results.info()
results.describe()

