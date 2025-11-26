
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os


os.makedirs('models', exist_ok=True)

# We Load datasets here
crime_df = pd.read_excel('static/data/Crime Rate Prediction dataset 1.xlsx')
pop_df = pd.read_excel('static/data/Crime Rate Prediction dataset 2.xlsx')


crime_df.columns = crime_df.columns.str.strip()
pop_df.columns = pop_df.columns.str.strip()


crime_df.rename(columns={
    'STATE': 'State',
    'Murder': 'MURDER',
    'Rape': 'RAPE',
    'Robbery': 'ROBBERY',
    'Kidnapping and Abduction': 'KIDNAPPING_AND_ABDUCTION'
}, inplace=True)

pop_df.rename(columns={
    'State/UT': 'State',
    'Population( in lakhs)': 'Population'
}, inplace=True)


print("Crime Dataset Columns:", crime_df.columns.tolist())
print("Population Dataset Columns:", pop_df.columns.tolist())


data = pd.merge(crime_df, pop_df, on=['State', 'Year'], how='inner')
data.dropna(inplace=True)


le_state = LabelEncoder()
data['StateEncoded'] = le_state.fit_transform(data['State'])


crime_types = ['MURDER', 'RAPE', 'ROBBERY', 'KIDNAPPING_AND_ABDUCTION']

for crime in crime_types:
    print(f"Training models for {crime}...")

    X = data[['StateEncoded', 'Year', 'Population']]
    y = data[crime]

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    joblib.dump(lr_model, f'models/lr_{crime}.pkl')

    #  Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=5)
    dt_model.fit(X, y)
    joblib.dump(dt_model, f'models/dt_{crime}.pkl')


joblib.dump(le_state, 'models/encoder_state.pkl')

print("All models trained and saved in 'models/' folder.")


def classify(rate):
    if rate < 2:
        return 'Low'
    elif rate < 5:
        return 'Moderate'
    else:
        return 'High'

def predict_crime_status(crime_type, state, year, population):
    lr_model = joblib.load(f'models/lr_{crime_type}.pkl')
    le_state = joblib.load('models/encoder_state.pkl')

    state_encoded = le_state.transform([state])[0]
    X = [[state_encoded, year, population]]

    predicted_cases = lr_model.predict(X)[0]
    crime_rate = predicted_cases / population
    crime_status = classify(crime_rate)

    return {
        'crime_type': crime_type,
        'state': state,
        'year': year,
        'population': population,
        'predicted_cases': round(predicted_cases),
        'crime_rate': round(crime_rate, 2),
        'crime_status': crime_status
    }
