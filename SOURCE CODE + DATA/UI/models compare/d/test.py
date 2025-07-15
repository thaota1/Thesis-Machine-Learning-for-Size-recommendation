import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

file_paths = {
    "fit_type": r"C:/Users/ADMIN/Downloads/d/model_eval_results.xlsx",
    "FIT": r"C:/Users/ADMIN/Downloads/d/FIT.xlsx",
    "LARGE": r"C:/Users/ADMIN/Downloads/d/LARGE.xlsx",
    "SMALL": r"C:/Users/ADMIN/Downloads/d/SMALL.xlsx"
}

frames = []
for cluster_name, path in file_paths.items():
    df = pd.read_excel(path)
    df["Cluster"] = cluster_name
    df["Label"] = df.apply(
        lambda row: f"height, weight, age" if row["Features Used"].strip().lower() in ["height, weight, age"]
        else f"height, weight, age  {row['Features Used'].strip()}", axis=1
    )
    df["Full Label"] = df["Cluster"].apply(lambda x: f"[{x}] ") + df["Label"]
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

st.title("Models Comparision")

st.sidebar.header("Model Selection")

# select clusters models
cluster_options = df["Cluster"].unique()
model_a_cluster = st.sidebar.selectbox("Select Cluster for Model A", cluster_options, key="cluster_a")

# feature options based on selected cluster
features_a = df[df["Cluster"] == model_a_cluster]["Label"].unique()
model1_label_raw = st.sidebar.selectbox("Select Features for Model A", features_a, key="m1")

model_b_cluster = st.sidebar.selectbox("Select Cluster for Model B", cluster_options, key="cluster_b")
features_b = df[df["Cluster"] == model_b_cluster]["Label"].unique()
model2_label_raw = st.sidebar.selectbox("Select Features for Model B", features_b, key="m2")

# display labels
model1_label = f"[{model_a_cluster}] {model1_label_raw}"
model2_label = f"[{model_b_cluster}] {model2_label_raw}"

# corresponding rows
model1 = df[(df["Cluster"] == model_a_cluster) & (df["Label"] == model1_label_raw)]
model2 = df[(df["Cluster"] == model_b_cluster) & (df["Label"] == model2_label_raw)]

if model1.empty or model2.empty:
    st.error("One or both model selections are invalid.")
else:
    row1 = model1.iloc[0]
    row2 = model2.iloc[0]

    st.subheader("Comparison Between Two Models")

    col1, col2 = st.columns(2)

    def highlight(val1, val2, better="lower"):
        if better == "lower":
            return "**:green[{}]**".format(val1) if val1 < val2 else "{}".format(val1)
        else:
            return "**:green[{}]**".format(val1) if val1 > val2 else "{}".format(val1)

    with col1:
        st.markdown(f"### Model A: {model1_label}")
        st.write("**Features:**", model1_label)
        st.write("**Cluster:**", row1["Cluster"])
        st.markdown("**MAE (Split):** " + highlight(row1["MAE (Split)"], row2["MAE (Split)"]))
        st.markdown("**RMSE (Split):** " + highlight(row1["RMSE (Split)"], row2["RMSE (Split)"]))
        st.markdown("**R2 (Split):** " + highlight(row1["R2 (Split)"], row2["R2 (Split)"], better="higher"))
        st.markdown("**MAE (Full):** " + highlight(row1["MAE (Full)"], row2["MAE (Full)"]))
        st.markdown("**RMSE (Full):** " + highlight(row1["RMSE (Full)"], row2["RMSE (Full)"]))
        st.markdown("**R2 (Full):** " + highlight(row1["R2 (Full)"], row2["R2 (Full)"], better="higher"))

    with col2:
        st.markdown(f"### Model B: {model2_label}")
        st.write("**Features:**", model2_label)
        st.write("**Cluster:**", row2["Cluster"])
        st.markdown("**MAE (Split):** " + highlight(row2["MAE (Split)"], row1["MAE (Split)"]))
        st.markdown("**RMSE (Split):** " + highlight(row2["RMSE (Split)"], row1["RMSE (Split)"]))
        st.markdown("**R2 (Split):** " + highlight(row2["R2 (Split)"], row1["R2 (Split)"], better="higher"))
        st.markdown("**MAE (Full):** " + highlight(row2["MAE (Full)"], row1["MAE (Full)"]))
        st.markdown("**RMSE (Full):** " + highlight(row2["RMSE (Full)"], row1["RMSE (Full)"]))
        st.markdown("**R2 (Full):** " + highlight(row2["R2 (Full)"], row1["R2 (Full)"], better="higher"))

    # comparison chart
    if st.checkbox("📉 Comparison Chart"):
        metric_names = ["MAE (Split)", "RMSE (Split)", "R2 (Split)", "MAE (Full)", "RMSE (Full)", "R2 (Full)"]
        values1 = [row1[m] for m in metric_names]
        values2 = [row2[m] for m in metric_names]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(metric_names))
        ax.bar([i - 0.2 for i in x], values1, width=0.4, label="Model A")
        ax.bar([i + 0.2 for i in x], values2, width=0.4, label="Model B")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        st.pyplot(fig)
