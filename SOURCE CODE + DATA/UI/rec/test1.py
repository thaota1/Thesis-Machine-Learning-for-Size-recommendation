import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# load data
rec_model = joblib.load("models/rec_size_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
kmeans = joblib.load("models/kmeans_model.pkl")
knn = joblib.load("models/knn_model.pkl")
tfidf_knn_vectors = joblib.load("models/tfidf_knn_vectors.pkl")
lookup_dict = joblib.load("models/fallback_lookup_dict.pkl")
df1 = pd.read_csv("df1_full.csv")
df1 = df1[df1['sent_score'] > 0].copy()  

st.set_page_config(page_title="Size Recommender", layout="centered")
st.title("Size Recommendation System")
st.write("Enter your information below to get a recommended clothing size.")

# input fields
col1, col2, col3 = st.columns(3)
with col1:
    height = st.text_input("Height (inches):", key="height")
with col2:
    weight = st.text_input("Weight (lbs):", key="weight")
with col3:
    age = st.text_input("Age:", key="age")

body_type = st.selectbox("Body Type:", sorted(df1['body_type'].dropna().unique()))
full_input = height.strip() != '' and weight.strip() != '' and age.strip() != ''

if full_input:
    fit_type = st.selectbox("Fit Type:", ['small', 'fit', 'large'])
else:
    fit_desc = st.text_input("How do you usually describe clothing fit? (e.g., 'tight at waist', 'very comfy'):", key="fit_desc")

submit = st.button("Submit")

if submit:
    st.markdown("---")

    if full_input:
        try:
            height = int(height)
            weight = int(weight)
            age = int(age)

            body_type_cat = df1['body_type'].astype('category')
            body_type_code = body_type_cat.cat.categories.get_loc(body_type)
            fit_map = {'small': 0, 'fit': 1, 'large': 2}
            fit_type_code = fit_map.get(fit_type.lower(), 1)

            input_df = pd.DataFrame([{
                'height': height,
                'weight': weight,
                'age': age,
                'body_type_encoded': body_type_code,
                'fit_type_encoded': fit_type_code
            }])

            predicted_size = round(rec_model.predict(input_df)[0])
            st.subheader(f"Recommended Size: {predicted_size}")

        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()

    else:
        try:
            if fit_desc.strip() == "":
                st.warning("Please describe clothing fit.")
                st.stop()

            if not hasattr(tfidf, 'vocabulary_'):
                st.error("Error in fallback prediction: tf-idf vector is not fitted properly.")
                st.stop()

            user_vec = tfidf.transform([fit_desc])
            predicted_cluster = kmeans.predict(user_vec)[0]
            cluster_to_fit = {0: 'fit', 1: 'small', 2: 'large'}
            fit_type = cluster_to_fit.get(predicted_cluster, 'fit')
            fallback = lookup_dict.get(body_type, {}).get(fit_type, None)

            if fallback:
                st.subheader(f"Recommended Size: {fallback}")
                predicted_size = fallback
            else:
                st.warning("No fallback size found for your body type and fit.")
                st.stop()

        except Exception as e:
            st.error(f"Error in fallback prediction: {e}")
            st.stop()

    # available products
    category_counts = df1['category'].value_counts()
    valid_categories = category_counts[category_counts > 1].index
    filtered_df = df1[(df1['rating'] > 5) & (df1['category'].isin(valid_categories))].copy()
    display_df = (
        filtered_df
        .sort_values('rating', ascending=False)
        .groupby('category')
        .apply(lambda x: x.sample(1))
        .reset_index(drop=True)
        .head(10)
    )
    display_df.index = range(len(display_df))
    st.markdown("\n**Here are some available items:**")
    st.dataframe(display_df[['item_id', 'category', 'review_summary']])

    st.session_state['display_df'] = display_df
    st.session_state['predicted_size'] = predicted_size
    st.session_state['show_results'] = True

# recommend similar items
if st.session_state.get('show_results'):
    index = st.text_input("Enter the index of the item from the list above to find similar items:", key="item_index")

    if index.strip().isdigit():
        index = int(index)
        display_df = st.session_state['display_df']
        if 0 <= index < len(display_df):
            selected_item = display_df.loc[index]
            selected_category = selected_item['category']
            selected_item_id = selected_item['item_id']

            similar_items = df1[
                (df1['category'] == selected_category) &
                (df1['item_id'] != selected_item_id)
            ][['item_id', 'category', 'review_summary', 'height', 'weight', 'age', 'rating']]

            if similar_items.empty:
                st.warning("No similar items found in the same category.")
            else:
                similar_items = similar_items.drop_duplicates(subset='item_id').head(5)
                similar_items['review_summary'] = similar_items['review_summary'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                st.markdown("\n**Recommended similar items:**")
                st.dataframe(similar_items.reset_index(drop=True))
        else:
            st.warning("Invalid index.")
    elif index:
        st.warning("Please enter a valid numeric index.")
