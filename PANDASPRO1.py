import pandas as pd
import numpy as np
# my first dataframe 
data={
    'Name':['Alice','Bob','Charlie','David','Eva'], 
    'Age':[24,27,22,32,29],
    'City':['New York','Los Angeles','Chicago','Houston','Phoenix'],
    "Ageg_group":[15,60,58,25,40]}
df=pd.DataFrame(data)
# print(df)
# print(df[["Name","City"]])
# print("the top row are\n",df.head(2))
# print("the bottom two rows are :\n",df.tail(2))
# print("the 1-2 rows and column 0-1:\n",df.iloc[1:3,0:2])
# print("the output based on the condition is:\n",df.iloc[1])
# print("the output based on the row\n",df.loc[2])
print("the selective data based on the condition :\n",df[df["Age"]>30])
df=pd.DataFrame(data)
sorted_df=df.sort_values(by="Age",ascending=False)
print("Dataframe sorted by Name:\n", sorted_df)