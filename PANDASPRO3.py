import numpy as np
import pandas as pd
df1=pd.DataFrame({
    "ID":[1,2,3,4],
    "Name":["AYA","MARIA","SARA","LINA"]
})
df2=pd.DataFrame({
    "ID":[2,5,6,7],
    "Score":[85,90,78,92]
})
merged_df_inner=pd.merge(df1,df2,on="ID",how="inner")
merged_df_outer=pd.merge(df1,df2,on="ID",how="outer")
merged_df_left=pd.merge(df1,df2,on="ID",how="left")
merged_df_right=pd.merge(df1,df2,on="ID",how="right")
# print("INNER JOIN:\n",merged_df_inner)
# print("OUTER JOIN:\n",merged_df_outer)
# print("RIGHT JOIN:\n",merged_df_right)
# print("LEFT JOIN:\n",merged_df_left)
deleted_updated=merged_df_outer.dropna()#[merged_df_outer["Name"]!="MARIA"]
print("Deleted and Updated Records:\n",deleted_updated)