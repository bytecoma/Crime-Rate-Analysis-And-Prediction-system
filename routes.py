
from flask import render_template, jsonify, request, send_file
import time
import joblib
import pandas as pd
import numpy as np

def register_routes(app):
    @app.context_processor
    def inject_time():
        return {'time': int(time.time())}

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')

    @app.route('/charts')
    def charts():
        return render_template('charts.html')

    @app.route('/data/<int:year>')
    def get_year_data(year):
        try:
            return send_file(f'static/data/fulldata{year}.json')
        except FileNotFoundError:
            return jsonify({"error": "Data not found"}), 404

    @app.route('/prediction')
    def prediction():
        return render_template('prediction.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        state = data['state'].strip().upper()
        crime_type = data['crime_type'].strip().upper()
        model_type = data.get('model', 'linear').lower()  
        year = 2000 + int(data['year'])

        
        pop_df = pd.read_excel('static/data/Crime Rate Prediction dataset 2.xlsx')
        pop_df.columns = pop_df.columns.str.strip()
        pop_df.rename(columns={'State/UT': 'State', 'Population( in lakhs)': 'Population'}, inplace=True)

        pop_row = pop_df[(pop_df['State'].str.upper() == state) & (pop_df['Year'] == year)]

        if pop_row.empty:
            state_pop = pop_df[pop_df['State'].str.upper() == state].sort_values('Year')
            if state_pop.empty:
                return jsonify({'error': 'No population data found for this state'}), 400
            latest = state_pop.iloc[-1]
            latest_year = latest['Year']
            latest_pop = float(latest['Population'])
            growth_rate = 0.015  
            years_diff = year - latest_year
            population = latest_pop * ((1 + growth_rate) ** years_diff)
        else:
            population = float(pop_row['Population'].values[0])

        try:
            le_state = joblib.load('models/encoder_state.pkl')
            state_encoded = le_state.transform([state])[0]

            if model_type == 'tree':
                model_path = f'models/dt_{crime_type}.pkl'
            else:
                model_path = f'models/lr_{crime_type}.pkl'

            model = joblib.load(model_path)

        except Exception as e:
            return jsonify({'error': f'Model or encoder load error: {str(e)}'}), 500

        
        X_input = np.array([[state_encoded, year, population]])
        predicted_cases = max(0, int(model.predict(X_input)[0]))
        crime_rate = round(predicted_cases / population, 2)

        def classify(rate):
            if rate < 2:
                return 'Low'
            elif rate < 5:
                return 'Moderate'
            else:
                return 'High'

        crime_status = classify(crime_rate)

        print("Prediction results:")
        print("Crime Status:", crime_status)
        print("Crime Rate:", crime_rate)
        print("Predicted Cases:", predicted_cases)
        print("Population:", population)

        return jsonify({
            'crime_status': crime_status,
            'crime_rate': f"{crime_rate}",
            'cases': predicted_cases,
            'population': f"{round(population, 2)} L"
        })