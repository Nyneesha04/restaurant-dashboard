import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Load dataset
df = pd.read_csv("data/cleaned_restaurant_ratings1_with_taste_quality.csv")

# Clean data
df["Area"] = df["Area"].str.strip().str.lower()
df["Food type"] = df["Food type"].str.strip().str.lower()
df["Area"] = df["Area"].str.replace("banjarahills", "banjara hills", case=False)

# Load model
model = joblib.load("models/restaurant_ratings_model.pkl")

# Page config
st.set_page_config(page_title="Restaurant Ratings Dashboard", layout="wide")
st.title("Restaurant Ratings Dashboard")

# Sidebar for filters
st.sidebar.header("Select Your Preferences")
area = st.sidebar.selectbox("Select Area", sorted(df["Area"].dropna().unique()))

# Filter food types based on selected area
food_df = df[df["Area"] == area]
food_types_set = set()
for ft in food_df["Food type"].dropna():
    food_types_set.update([f.strip() for f in ft.split(",")])
food_types = sorted(food_types_set)
food_type = st.sidebar.selectbox("Select Food Type", food_types)

min_rating = st.sidebar.slider("Minimum Avg Rating", 0.0, 5.0, 3.0, 0.1)
max_price = st.sidebar.slider("Maximum Price", int(df["Price"].min()), int(df["Price"].max()), 800)

if st.sidebar.button("Show Restaurants"):
    filtered_df = df[
        (df["Area"] == area) &
        (df["Food type"].str.contains(food_type, case=False, na=False)) &
        (df["Avg ratings"] >= min_rating) &
        (df["Price"] <= max_price)
    ].copy()

    if filtered_df.empty:
        st.warning("No restaurants match your filters.")
    else:
        # ML prediction
        filtered_df["log_total_ratings"] = np.log1p(filtered_df["Total ratings"])
        filtered_df["interaction"] = filtered_df["log_total_ratings"] * filtered_df["Delivery time"]
        X = filtered_df[["log_total_ratings", "Delivery time", "interaction"]]
        filtered_df["Predicted Rating"] = model.predict(X)

        top_df = filtered_df.sort_values(by="Predicted Rating", ascending=False).head(5)

        st.subheader("Top 5 Recommended Restaurants")
        st.dataframe(top_df[["Restaurant", "Price", "Avg ratings", "Predicted Rating", "Taste", "Quality", "Delivery time"]])

        # Bar chart
        st.subheader("Top Restaurants: Predicted Ratings vs Delivery Time")
        fig_bar = px.bar(
            top_df,
            x="Restaurant",
            y="Predicted Rating",
            color="Delivery time",
            hover_data=["Price", "Taste", "Quality", "Avg ratings", "Food type"],
            text="Predicted Rating",
            height=450
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter plot
        st.subheader("Price vs Predicted Rating")
        fig_scatter = px.scatter(
            top_df,
            x="Price",
            y="Predicted Rating",
            size="Total ratings",
            color="Taste",
            hover_name="Restaurant",
            hover_data=["Quality", "Area", "Food type", "Delivery time"],
            size_max=40,
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.subheader("Bubble Chart: Price vs Delivery Time with Predicted Ratings")

        fig_bubble = px.scatter(
            top_df,
            x="Price",
            y="Delivery time",
            size="Total ratings",
            color="Predicted Rating",
            hover_name="Restaurant",
            hover_data=["Taste", "Quality", "Food type"],
            size_max=50,
            height=500,
            title="Bubble Chart: Delivery Time vs Price"
        )

        st.plotly_chart(fig_bubble, use_container_width=True)
else:
    st.info("Please select your preferences from the sidebar and click 'Show Restaurants'.")