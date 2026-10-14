from pathlib import Path
import pandas as pd
import numpy as np
reviews_df = pd.read_csv('Zomato Restaurant reviews.csv')
metadata_df = pd.read_csv('Zomato Restaurant names and Metadata.csv')
Path('models').mkdir(exist_ok=True)
# Write your code to make your dataset analysis ready.

reviews_clean = reviews_df.drop_duplicates().copy()
metadata_clean = metadata_df.copy()

# Convert Rating from object/string to numeric.
reviews_clean["Rating_num"] = pd.to_numeric(reviews_clean["Rating"], errors="coerce")

# Convert Cost from values like "1,300" to numeric 1300.
metadata_clean["Cost_num"] = pd.to_numeric(
    metadata_clean["Cost"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Basic missing value handling.
reviews_clean["Review"] = reviews_clean["Review"].fillna("").astype(str).str.strip()
reviews_clean["Reviewer"] = reviews_clean["Reviewer"].fillna("Unknown")
reviews_clean["Metadata"] = reviews_clean["Metadata"].fillna("Unknown")
reviews_clean["Time"] = pd.to_datetime(reviews_clean["Time"], errors="coerce")

metadata_clean["Collections"] = metadata_clean["Collections"].fillna("Not Available")
metadata_clean["Timings"] = metadata_clean["Timings"].fillna("Not Available")

# Remove rows that cannot be used for sentiment analysis.
reviews_clean = reviews_clean[
    reviews_clean["Rating_num"].notna() & (reviews_clean["Review"] != "")
].copy()

# Convert rating into sentiment labels.
def sentiment_from_rating(rating):
    if rating <= 2:
        return "Negative"
    elif rating <= 3.5:
        return "Neutral"
    else:
        return "Positive"

reviews_clean["Sentiment"] = reviews_clean["Rating_num"].apply(sentiment_from_rating)

# Create useful features for later EDA and ML.
reviews_clean["Review_Length"] = reviews_clean["Review"].str.split().str.len()
reviews_clean["Has_Pictures"] = np.where(reviews_clean["Pictures"] > 0, 1, 0)

metadata_clean["Primary_Cuisine"] = (
    metadata_clean["Cuisines"]
    .fillna("Unknown")
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)

# Merge review data with restaurant metadata.
zomato_df = reviews_clean.merge(
    metadata_clean,
    left_on="Restaurant",
    right_on="Name",
    how="left"
)

# Save cleaned data for the next steps.
Path("data").mkdir(exist_ok=True)
zomato_df.to_csv("data/zomato_cleaned_step1.csv", index=False)

print("Cleaned merged dataset shape:", zomato_df.shape)
print("Cleaned file saved at: data/zomato_cleaned_step1.csv")
print(zomato_df.head())

