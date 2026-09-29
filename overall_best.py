import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd


df = pd.read_csv("results.csv")

df_min = df[df.groupby(['category', 'error_type'])['image_auroc'].transform(min) == df['image_auroc']]
df = df[df.groupby(['category', 'error_type'])['image_auroc'].transform(max) == df['image_auroc']]

category_order = (
    df.groupby("category")["image_auroc"]
      .max()
      .sort_values(ascending=False)
      .index
)

df["category"] = pd.Categorical(
    df["category"],
    categories=category_order,
    ordered=True,
)

df = df.sort_values(
    ["category", "image_auroc"],
    ascending=[True, False]
)

im_auroc_fig, ax = plt.subplots(figsize=(12,15))    

colors = ["#4875ff",
            "#63bf2d",
            "#881c9d",
            "#c49b00",
            "#2c4ab2",
            "#f27f0c",
            "#6eb9ff",
            "#7f5d00",
            "#ff6ee5",
            "#1d6009",
            "#d0006a",
            "#019c66",
            "#9c2146",
            "#739463",
            "#ff8098"]


duplicates_min = df_min["error_type/prompt"].duplicated(keep=False)

df_min.loc[duplicates_min, "error_type/prompt"] = (
    df_min.loc[duplicates_min, "category"].astype(str) + ": " + df.loc[duplicates_min, "error_type/prompt"]
)

duplicates = df["error_type/prompt"].duplicated(keep=False)


df.loc[duplicates, "error_type/prompt"] = (
    df.loc[duplicates, "category"].astype(str) + ": " + df.loc[duplicates, "error_type/prompt"]
)

p = sns.barplot(x="image_auroc", y="error_type/prompt", data=df, hue="category", dodge=False, palette=colors)

sns.scatterplot(
    data=df_min,
    y="error_type/prompt",
    x="image_auroc",
    color="black",
    marker=".",
    s=60,
    zorder=3,
    ax=ax,
)

handles, labels = ax.get_legend_handles_labels()

handles.append(
    Line2D(
        [0],
        [0],
        marker=".",
        color="w",
        markerfacecolor="black",
        markersize=8,
        label="worst score"
    )
)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )
im_auroc_fig.set_layout_engine("tight")
plt.title("Best Image-AUROC Scores per Errortype")
plt.ylabel("Errortype/Prompt")
plt.xlabel("AUROC Score")
plt.xticks(size="small")
ax.legend(handles=handles, title="Category")
ax.set_ylim(len(df), -1)
plt.savefig("figures/best.png")


