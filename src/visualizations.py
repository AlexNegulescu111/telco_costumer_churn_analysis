def distribution_count(column, column_name):
    from utils import ROOT
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    data = column.value_counts().sort_values(ascending=False)
    if len(data) > 20:
        data = data.head(20)

    height = max(0.6 * len(data), 2.5)
    plt.figure(figsize=(8, height), dpi=100)

    sns.barplot(x=data.values, y=data.index, color="blue", orient = "h")
    plt.title(f"Distribution of {column_name}")
    plt.xlabel("Frequency")
    plt.ylabel(column_name)
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(str(ROOT)+f"/plots/distribution_of_{column_name}.png", dpi=100, bbox_inches="tight")
    plt.show()

# Function to create grouped churn plots for a given category
def plot_churn_by_group(var_list, title_prefix, df):
    from utils import ROOT
    import seaborn as sns
    import matplotlib.pyplot as plt
    n = len(var_list)
    cols = 2
    rows = (n + 1) // cols

    plt.figure(figsize=(12, 4 * rows))
    for i, col in enumerate(var_list, 1):
        plt.subplot(rows, cols, i)
        sns.countplot(data=df, x=col, hue="churn")
        plt.title(f"{title_prefix}: {col}")
        plt.ylabel("Number of Customers")
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.suptitle(f"Churn based on variables in category: {title_prefix}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(str(ROOT)+f"/plots/{title_prefix.replace(" ", "_").lower()}.png", dpi=100, bbox_inches="tight")
    plt.show()
