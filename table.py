import pandas as pd
import dataframe_image as dfi


df = pd.read_csv("results.csv")

errortype_rows = (
    df.groupby(["category", "error_type"])
      .agg(
          nr_of_prompts=("prompt", "size"),
          prompts=("prompt", list)
      )
      .reset_index()
)

category_rows = (
    df.groupby("category")
      .size()
      .reset_index(name="nr_of_prompts")
)

category_rows["error_type"] = None
category_rows["prompts"] = None

category_rows = category_rows[
    ["category", "error_type", "nr_of_prompts", "prompts"]
]

df = pd.concat([category_rows, errortype_rows], ignore_index=True)

category_order = df["category"].drop_duplicates().tolist()

df["category"] = pd.Categorical(
    df["category"],
    categories=category_order,
    ordered=True
)

df = (
    df
    .sort_values(["category"])
    .reset_index(drop=True)
)

print(df.to_latex(index=False,))

dfi.export(df, "figures/table.png")

