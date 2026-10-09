import pandas as pd

print("Pandas version:", pd.__version__)

data = {
    "Player": ["LeBron James", "Stephen Curry", "Kevin Durant"],
    "Points": [2, 28, 27],
    "Assists": [1, 6, 5]
}

df = pd.DataFrame(data)

print(df)