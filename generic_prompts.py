import dataframe_image as dfi
import pandas as pd

df = pd.read_csv("results.csv")

df = df[df["prompt"].isin(["damage", "damaged", "broken"])]

df = df.sort_values(
    ["image_auroc"],
    ascending=[False]
)

dfi.export(df[["category", "error_type", "prompt", "image_auroc"]].iloc[:25].style.hide(), "figures/generics.png")

comparison = (
    df[df["Prompt"].isin(["damaged", "damage"])]
    .pivot(index="Category/Errortype", columns="Prompt", values="image_auroc")
    .sort_values("damaged", ascending=False)
)


columns = ["Prompt", "0.5", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]", "(0.9, 1]"]
data = []
for prompt in ["broken", "damage", "damaged"]:
    row = [prompt]
    gen_df = df[df["prompt"] == prompt]
    row.append(((df["image_auroc"] <= 0.50) & (df["Prompt"] == prompt)).sum())
    row.append(((0.50 < df["image_auroc"]) & (df["image_auroc"] <= 0.60) & (df["prompt"] == prompt)).sum())
    row.append(((0.60 < df["image_auroc"]) & (df["image_auroc"] <= 0.70) & (df["prompt"] == prompt)).sum())
    row.append(((0.70 < df["image_auroc"]) & (df["image_auroc"] <= 0.80) & (df["prompt"] == prompt)).sum())
    row.append(((0.80 < df["image_auroc"]) & (df["image_auroc"] <= 0.90) & (df["prompt"] == prompt)).sum())
    row.append(((df["image_auroc"] > 0.90) & (df["prompt"] == prompt)).sum())
    data.append(row)


df_2 = pd.DataFrame(data=data, columns=columns)
print(df_2.to_latex(index=False,
                  float_format="{:.2f}".format,
                  ))

dfi.export(df_2, "figures/generics_counts.png")


