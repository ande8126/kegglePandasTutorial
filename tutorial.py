import pandas as pd
from data.nbaDataImport import load_nba_games

df = load_nba_games()

print("Pandas version:", pd.__version__)

# data = {
#     "Player": ["LeBron James", "Stephen Curry", "Kevin Durant"],
#     "Points": [2, 28, 27],
#     "Assists": [1, 6, 5]
# }

print(df.head())
print(df.shape)