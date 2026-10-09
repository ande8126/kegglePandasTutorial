import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def load_nba_games():
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )

    query = """
    SELECT
        id,
        game_date,
        season,
        home_team_id,
        home_team_name,
        visitor_team_id,
        visitor_team_name,
        home_team_score,
        visitor_team_score,
        postseason
    FROM bdl_nba_games
    WHERE home_team_score IS NOT NULL
    AND visitor_team_score IS NOT NULL
    AND home_team_score <> 0
    AND visitor_team_score <> 0;
    """

    df = pd.read_sql(query, engine)

    print(df.head())
    print(df.shape)

    return df

