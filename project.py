import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('Anime_ratings.csv')
 # --- Fix basic types ---
# Rank, Episodes, Episode length are object now
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df["Episodes"] = pd.to_numeric(df["Episodes"].replace("Unknown", np.nan), errors="coerce")

# You can ignore Episode length for now (text like "24 min. per ep.")
# Or later engineer a numeric minutes column from it.

# --- Extract Season & Year from Release Date ---
df["Release Date"] = df["Release Date"].fillna("Unknown")

split_cols = df["Release Date"].str.split(" ", n=1, expand=True)
df["Season"] = split_cols[0]
df["Year_str"] = split_cols[1]

valid_seasons = ["Winter", "Spring", "Summer", "Fall"]
df["Season"] = df["Season"].where(df["Season"].isin(valid_seasons), "Unknown")

df["Year"] = pd.to_numeric(df["Year_str"], errors="coerce").astype("Int64")
df = df.drop(columns=["Year_str"])

# --- Simple target: High vs Low popularity (classification) ---
# Lower Popularity number = more popular, so invert it
pop_cut = df["Popularity"].quantile(0.33)
df["popular_label"] = (df["Popularity"] <= pop_cut).astype(int)  # 1 = high popular

# --- Choose features (X) and target (y) ---
feature_cols = ["Rank", "Score", "Episodes", "Year"]
X = df[feature_cols].copy()
y = df["popular_label"]

# Handle missing values: fill numeric with median
for c in feature_cols:
    X[c] = X[c].fillna(X[c].median())

#####
######
#########
# Distribution of score
sns.histplot(df["Score"].dropna(), bins=20)
plt.title("Score distribution")
plt.show()

# Popularity vs Score
sns.scatterplot(data=df, x="Score", y="Popularity")
plt.title("Popularity vs Score (lower = more popular)")
plt.show()

# Average score by Season
sns.barplot(data=df.dropna(subset=["Season","Score"]),
            x="Season", y="Score", estimator="mean")
plt.title("Average score by season")
plt.show()


#####3
######
########
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


###
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



# Example new anime data (dummy values)
new_anime = pd.DataFrame([{
    "Rank": 500.0,
    "Score": 8.2,
    "Episodes": 24,
    "Year": 2022
}])

# Make sure columns match and fill if needed
for c in feature_cols:
    if c not in new_anime.columns:
        new_anime[c] = X[c].median()

pred_label = clf.predict(new_anime)[0]
print("Predicted popularity class:", pred_label)  # 1 = high popular, 0 = low