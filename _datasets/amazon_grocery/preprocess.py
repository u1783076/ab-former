import pandas as pd

ratings = pd.read_csv("ratings_Grocery_and_Gourmet_Food.csv", header=None)
ratings.columns = ["user_id","item_id","rating","timestamp"]
ratings = ratings[ratings.rating >= 4]
ratings["user_id"] = ratings["user_id"].astype("category")
ratings["item_id"] = ratings["item_id"].astype("category")

df = pd.read_feather("item_text_descriptions.feather")
df = df.drop_duplicates("item_id")
df = df[df.item_id.isin(ratings.item_id)]
df.to_feather("item_text_descriptions.feather")

ratings = ratings[ratings.item_id.isin(df.item_id)]
ratings.to_feather("ratings.feather")
