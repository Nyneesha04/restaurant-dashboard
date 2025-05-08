from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("models/restaurant_ratings_model.pkl")  # Make sure path is correct

# Load dataset
data = pd.read_csv("data/cleaned_restaurant_ratings1.csv")  # Update filename if different

# Extract dropdown values
areas = sorted(data['Area'].dropna().unique())
restaurants = sorted(data['Restaurant'].dropna().unique())
prices = sorted(data['Price'].dropna().unique())
food_types = sorted(data['Food type'].dropna().unique())
ratings = sorted(data['Avg ratings'].dropna().unique())
delivery_times = sorted(data['Delivery time'].dropna().unique())

@app.route('/')
def home():
    return render_template(
        'index.html',
        areas=areas,
        restaurants=restaurants,
        prices=prices,
        food_types=food_types,
        ratings=ratings,
        delivery_times=delivery_times
    )

@app.route('/predict', methods=['POST'])
def predict():
    selected_area = request.form['area']
    selected_restaurant = request.form['restaurant']
    selected_price = float(request.form['price'])
    selected_food_type = request.form['food_type']
    selected_rating = float(request.form['rating'])
    selected_delivery_time = int(request.form['delivery_time'])

    # Make sure to match the structure your model was trained with!
    input_data = [[
        selected_price,
        selected_rating,
        selected_delivery_time
        # You can also add food_type encoding if your model needs it
    ]]

    prediction = model.predict(input_data)

    return render_template(
        'index.html',
        areas=areas,
        restaurants=restaurants,
        prices=prices,
        food_types=food_types,
        ratings=ratings,
        delivery_times=delivery_times,
        prediction_text=f"Predicted Rating: {prediction[0]:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)