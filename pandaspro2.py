import numpy as np
import pandas as pd
data={
    "Product":["A","B","C","A","B"],
    "Region":["North","South","East","West","Central"],
    "Sales":[250,150,400,300,500],
    "unit":[5,3,8,6,10]
}
df=pd.DataFrame(data)
groped_df=df.groupby(["Product"])["Sales"].max()
print("Grouped DataFrame by Region with sum:\n",groped_df)
agg=df.groupby("Product").agg({
    "Sales":["sum","mean","max","min"],
    "unit":["sum","mean","max","min"]
})
print("Aggregated :\n",agg)