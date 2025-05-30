import requests
import pandas as pd
import numpy as np
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import random
import joblib
from datetime import datetime, timedelta
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.db.utils import IntegrityError
from .models import Prediction
from .serializers import PredictionSerializer

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Prediction
from .serializers import PredictionSerializer
from userapp.models import CustomUser
from django.core.mail import send_mail


# Function to get weather data
def get_weather_data(location):
    api_key = "54bfe931d3e776f190416f2bd20819d3"  # Use your OpenWeatherMap API key

    if not location:
        raise ValueError("Location must be specified")
    
    # Construct the API URL
    try:
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        raise Exception(f"Error fetching weather data: {e}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Convert temperature from Kelvin to Celsius
        temp_celsius = data['main']['temp'] - 273.15
        
        # Extract relevant weather information
        weather_info = {
            'temperature': round(temp_celsius, 2),
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'rainfall': data['rain']['1h'] if 'rain' in data and '1h' in data['rain'] else 0
        }
        # Also return the latitude and longitude from the 'coord' field
        weather_info['latitude'] = data['coord']['lat']
        weather_info['longitude'] = data['coord']['lon']
        
        print(f"Weather Data: {weather_info}\n")
        
        return weather_info
    else:
        print(f"Error fetching weather data: {response.status_code}")
        raise Exception(f"Error fetching weather data: {response.status_code}")





def predict_soil(district, latitude, longitude):

    try:
        # Load the saved model and preprocessor
        soil_model = load('../Rwanda soil data/best_soil_texture_model.joblib')  # Correct path to the soil texture model
        # Load the original dataset for fitting the preprocessor
        dataset = pd.read_csv('../Rwanda soil data/rwanda_complete_districts.csv')  # Use your original dataset
        
        # Update features and targets for soil texture prediction
        X_filtered = dataset[['District', 'Latitude', 'Longitude']]  # Features used during training

        # Define the preprocessor (fitted on the training data)
        preprocessor = ColumnTransformer(
            transformers=[('district', OneHotEncoder(), ['District'])],
            remainder='passthrough'  # Keep Latitude and Longitude as they are
        )

        # Fit the preprocessor on the training data (X_filtered)
        preprocessor.fit(X_filtered)

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'District': [district],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })

        # Preprocess the input data
        processed_input = preprocessor.transform(input_data)

        # Use the model to predict the soil texture
        predicted_soil = soil_model.predict(processed_input)  # Using the correct model variable
        print(f"Predicted soil type: {predict_soil}\n")

        # Return the predicted soil texture
        return predicted_soil[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def generate_random_soil_data():
    """
    Generate random soil and environmental data within realistic ranges
    """
    return {
        'N': random.uniform(0, 140),  # Nitrogen (mg/kg)
        'P': random.uniform(0, 80),   # Phosphorus (mg/kg)
        'K': random.uniform(0, 100),  # Potassium (mg/kg)
        'ph': random.uniform(5.5, 7.0),  # Soil pH
        'elevation': random.uniform(1300, 2000),  # Elevation (m)
        'slope': random.uniform(2, 18),  # Slope (degrees)
        'aspect': random.uniform(0, 360),  # Aspect (degrees)
        'water_holding_capacity': random.uniform(0.55, 0.85),  # Water holding capacity
        'solar_radiation': random.uniform(13, 19),  # Solar radiation (MJ/m²/day)
        'ec': random.uniform(0.2, 0.6),  # Electrical conductivity (dS/m)
        'zn': random.uniform(1.0, 3.0)  # Zinc content (mg/kg)
    }

def preprocess_input_data(input_data, soil_texture):
    """
    Preprocess input data to match the format expected by the model
    """
    # Convert input data to DataFrame format
    input_dict = {k: [v] for k, v in input_data.items()}
    df = pd.DataFrame(input_dict)
    
    # Create dummy variables for soil texture
    soil_textures = ['Clay Loam', 'Loam', 'Sandy', 'Sandy Clay', 'Sandy Loam']
    for texture in soil_textures:
        column_name = f'soil_texture_{texture}'
        df[column_name] = 1 if texture == soil_texture else 0
    
    # Load the scaler to get the exact feature names
    scaler = load('../crop recommendation/scaler.joblib')
    expected_features = scaler.feature_names_in_
    
    # Ensure columns are in the same order as during training
    df = df.reindex(columns=expected_features, fill_value=0)
    
    return df

def predict_crop(location, api_key, soil_texture):
    """
    Make crop predictions using weather data and random soil parameters
    
    Parameters:
    location (str): Location name for weather data
    api_key (str): OpenWeatherMap API key
    soil_texture (str): Soil texture value
    
    Returns:
    dict: Prediction results and input conditions
    """
    try:
        # Get weather data
        weather_data = get_weather_data(location)
        
        # Generate random soil data
        soil_data = generate_random_soil_data()
        
        # Combine weather and soil data
        input_data = {**weather_data, **soil_data}
        
        # Load the saved models and preprocessors
        model = load('../crop recommendation/decision_tree_crop_recommendation_model.joblib')
        scaler = load('../crop recommendation/scaler.joblib')
        label_encoder = load('../crop recommendation/label_encoder.joblib')
        
        # Preprocess the input data
        input_df = preprocess_input_data(input_data, soil_texture)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Make prediction
        prediction = model.predict(input_scaled_df)
        
        # Convert numeric prediction back to crop name
        crop_name = label_encoder.inverse_transform(prediction)
        
        return {
            'status': 'success',
            'predicted_crop': crop_name[0],
            'weather_data': weather_data,
            'soil_data': soil_data,
            'input_conditions': {
                'soil_texture': soil_texture,
                **input_data
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'traceback': str(e.__traceback__)
        }


import os
import joblib
import pandas as pd

def predict_water_requirement(location, weather_data, soil_data, crop_name):
    """
    Predict the water requirement for a given crop based on weather and soil data.
    
    Parameters:
        weather_data (dict): Dictionary containing weather data.
        soil_data (dict): Dictionary containing soil data.
        crop_name (str): Name of the predicted crop.
        
    Returns:
        float: Predicted water requirement in mm/day.
    """
    try:
        # Multiple possible paths to check for the model file
        possible_paths = [
            '../Crop water requirement/best_model_xgboost.joblib',
            './Crop water requirement/best_model_xgboost.joblib',
            'Crop water requirement/best_model_xgboost.joblib',
            './best_model_xgboost.joblib',
            'best_model_xgboost.joblib',
            os.path.join(os.path.dirname(__file__), '..', 'Crop water requirement', 'best_model_xgboost.joblib'),
            os.path.join(os.path.dirname(__file__), 'Crop water requirement', 'best_model_xgboost.joblib'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Crop water requirement', 'best_model_xgboost.joblib')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {path}")
                break
        
        if model_path is None:
            # List current directory contents to help debug
            current_dir = os.getcwd()
            print(f"Current working directory: {current_dir}")
            print(f"Contents of current directory: {os.listdir(current_dir)}")
            
            # Check parent directory if it exists
            parent_dir = os.path.dirname(current_dir)
            if os.path.exists(parent_dir):
                print(f"Contents of parent directory ({parent_dir}): {os.listdir(parent_dir)}")
            
            raise FileNotFoundError(
                f"Model file 'best_model_xgboost.joblib' not found in any of the expected locations: {possible_paths}. "
                f"Please ensure the model file exists and update the path accordingly."
            )
        
        # Load the saved model
        model_data = joblib.load(model_path)
        scaler = model_data['scaler']
        model = model_data['model']
        feature_names = model_data['feature_names']
        label_encoder = model_data['label_encoder']
        
        print(f"Model loaded successfully. Feature names: {feature_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create a comprehensive test data dictionary
    test_data = {}
    
    # Add weather and soil data
    test_data.update(weather_data)
    test_data.update(soil_data)
    
    # Map crop name to encoded value
    crop_mapping = {
        'Maize': 0, 'Rice': 1, 'Sorghum': 2, 'Wheat': 3, 'Millet': 4,
        'Cassava': 5, 'Sweet Potatoes': 6, 'Irish Potatoes': 7, 'Yams': 8, 'Taro': 9,
        'Beans': 10, 'Soybeans': 11, 'Groundnuts': 12, 'Peas': 13, 'Green Grams': 14,
        'Coffee': 15, 'Tea': 16, 'Pyrethrum': 17, 'Sugarcane': 18, 'Cotton': 19,
        'Banana': 20, 'Avocado': 21, 'Mango': 22, 'Pineapple': 23, 'Passion Fruit': 24, 
        'Tree Tomato': 25, 'Tomatoes': 26, 'Cabbage': 27, 'Carrots': 28, 'Onions': 29, 
        'Green Peppers': 30, 'Eggplant': 31, 'Sunflower': 32, 'Palm Oil': 33, 
        'Macadamia': 34, 'Ginger': 35, 'Chili Peppers': 36, 'Vanilla': 37
    }
    
    # Encode crop
    if crop_name in crop_mapping:
        test_data['crop_encoded'] = crop_mapping[crop_name]
    else:
        print(f"Warning: Crop '{crop_name}' not found in mapping. Using default (0).")
        test_data['crop_encoded'] = 0
    
    # Determine and encode soil type based on soil properties
    def determine_soil_type(soil_data):
        # Simple heuristic based on soil properties
        # You may need to adjust this based on your actual soil classification logic
        ph = soil_data.get('ph', 7.0)
        if ph < 5.5:
            return 'sandy'  # Acidic soils are often sandy
        elif ph > 7.5:
            return 'clay'   # Alkaline soils are often clay
        else:
            return 'loamy'  # Neutral pH often indicates loamy soil
    
    soil_type_mapping = {'sandy': 0, 'loamy': 1, 'clay': 2, 'silty': 3, 'peaty': 4}
    soil_type = determine_soil_type(soil_data)
    test_data['soil_type_encoded'] = soil_type_mapping[soil_type]
    
    # Determine slope category
    def determine_slope_category(slope_value):
        if slope_value <= 5:
            return '0-5'
        elif slope_value <= 10:
            return '5-10'
        elif slope_value <= 15:
            return '10-15'
        else:
            return '>15'
    
    slope_mapping = {'0-5': 0, '5-10': 1, '10-15': 2, '>15': 3}
    slope_category = determine_slope_category(soil_data.get('slope', 5))
    test_data['slope_encoded'] = slope_mapping[slope_category]
    
    # Determine temperature category
    def determine_temp_category(temperature):
        if temperature < 15:
            return '<15'
        elif temperature <= 25:
            return '15-25'
        elif temperature <= 30:
            return '25-30'
        else:
            return '>30'
    
    temp_mapping = {'<15': 0, '15-25': 1, '25-30': 2, '>30': 3}
    temp_category = determine_temp_category(weather_data.get('temperature', 20))
    test_data['temp_encoded'] = temp_mapping[temp_category]
    
    # Determine humidity category
    def determine_humidity_category(humidity):
        if humidity < 40:
            return '<40'
        elif humidity <= 60:
            return '40-60'
        elif humidity <= 80:
            return '60-80'
        else:
            return '>80'
    
    humidity_mapping = {'<40': 0, '40-60': 1, '60-80': 2, '>80': 3}
    humidity_category = determine_humidity_category(weather_data.get('humidity', 60))
    test_data['humidity_encoded'] = humidity_mapping[humidity_category]
    
    # Determine wind speed category
    def determine_wind_category(wind_speed):
        if wind_speed < 2:
            return '<2'
        elif wind_speed <= 5:
            return '2-5'
        elif wind_speed <= 8:
            return '5-8'
        else:
            return '>8'
    
    wind_mapping = {'<2': 0, '2-5': 1, '5-8': 2, '>8': 3}
    wind_category = determine_wind_category(weather_data.get('wind_speed', 3))
    test_data['wind_speed_encoded'] = wind_mapping[wind_category]
    
    # Set default growth stage (you can make this dynamic based on planting date)
    growth_stage_mapping = {'germination': 0, 'vegetative': 1, 'flowering': 2, 'fruit_development': 3, 'maturity': 4}
    test_data['growth_stage_encoded'] = growth_stage_mapping['vegetative']  # Default to vegetative
    
    # Add base water requirements (these should match your training data structure)
    # You'll need to add the actual base water requirement values here
    # For now, using placeholder values - replace with actual lookup based on crop and conditions
    elevation = soil_data.get('elevation', 1500)
    
    if elevation < 1500:
        # Low altitude
        test_data['low_altitude_dry'] = 600  # These should be looked up from your data
        test_data['low_altitude_rainy'] = 420
        test_data['mid_altitude_dry'] = 0
        test_data['mid_altitude_rainy'] = 0
        test_data['high_altitude_dry'] = 0
        test_data['high_altitude_rainy'] = 0
    elif elevation < 2000:
        # Mid altitude
        test_data['low_altitude_dry'] = 0
        test_data['low_altitude_rainy'] = 0
        test_data['mid_altitude_dry'] = 550
        test_data['mid_altitude_rainy'] = 385
        test_data['high_altitude_dry'] = 0
        test_data['high_altitude_rainy'] = 0
    else:
        # High altitude
        test_data['low_altitude_dry'] = 0
        test_data['low_altitude_rainy'] = 0
        test_data['mid_altitude_dry'] = 0
        test_data['mid_altitude_rainy'] = 0
        test_data['high_altitude_dry'] = 500
        test_data['high_altitude_rainy'] = 350
    
    print(f"Test data prepared: {test_data}")
    
    # Create DataFrame and align with training features
    df_test = pd.DataFrame([test_data])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df_test.columns:
            df_test[feature] = 0  # Fill missing features with 0
    
    # Select only the features used in training
    df_test = df_test[feature_names]
    
    print(f"Test DataFrame shape: {df_test.shape}")
    print(f"Test DataFrame columns: {df_test.columns.tolist()}")
    print(f"Test DataFrame values: {df_test.iloc[0].to_dict()}")
    
    # Scale the features (only for models that require scaling)
    # Check if the model needs scaling by looking at the model type
    model_name = str(type(model).__name__)
    if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
        X_test_scaled = scaler.transform(df_test)
        print("Applied scaling for linear model")
    else:
        X_test_scaled = df_test.values
        print("No scaling applied for tree-based model")
    
    # Predict water requirement
    predicted_water_requirement = model.predict(X_test_scaled)
    
    print(f"Raw prediction: {predicted_water_requirement}")
    
    return predicted_water_requirement[0]


# Alternative function with explicit model path parameter
def predict_water_requirement_with_path(location, weather_data, soil_data, crop_name, model_path):
    """
    Predict the water requirement for a given crop based on weather and soil data.
    This version allows you to specify the exact model path.
    
    Parameters:
        location: Location data
        weather_data (dict): Dictionary containing weather data.
        soil_data (dict): Dictionary containing soil data.
        crop_name (str): Name of the predicted crop.
        model_path (str): Exact path to the model file.
        
    Returns:
        float: Predicted water requirement in mm/day.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load the saved model
        model_data = joblib.load(model_path)
        scaler = model_data['scaler']
        model = model_data['model']
        feature_names = model_data['feature_names']
        label_encoder = model_data['label_encoder']
        
        print(f"Model loaded successfully from: {model_path}")
        print(f"Feature names: {feature_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Rest of the function remains the same...
    # [Include all the same processing logic as the original function]
    # This is just showing the model loading part with explicit path



def predict_crop_and_water_requirement(location, api_key):
    
    """
    Predict crop and water requirement based on weather data, soil data, and the crop.
    
    Parameters:
        location (str): Location name for weather data.
        api_key (str): OpenWeatherMap API key.
    
    Returns:
        dict: Prediction results including crop and water requirement.
    """
    try:
        # Step 1: Get weather data
        weather_data = get_weather_data(location)
        
        # Step 2: Generate random soil data
        soil_data = generate_random_soil_data()
        
        # Step 3: Predict soil texture
        latitude = weather_data['latitude']
        longitude = weather_data['longitude']
        soil_texture = predict_soil(location, latitude, longitude)
        
        # Step 4: Combine weather and soil data
        input_data = {**weather_data, **soil_data}
        
        # Step 5: Load models and preprocessors
        model = load('../crop recommendation/decision_tree_crop_recommendation_model.joblib')
        scaler = load('../crop recommendation/scaler.joblib')
        label_encoder = load('../crop recommendation/label_encoder.joblib')
        
        # Step 6: Preprocess the input data for crop prediction
        input_df = preprocess_input_data(input_data, soil_texture)
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Step 7: Predict crop
        prediction = model.predict(input_scaled_df)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        # Step 8: Predict water requirement
        predicted_water_requirement = predict_water_requirement(
            location=location,
            weather_data=weather_data,
            soil_data=soil_data,
            crop_name=crop_name
        )
        
        # Step 9: Return the combined results
        return {
            'status': 'success',
            'predicted_crop': crop_name,
            'predicted_water_requirement': round(predicted_water_requirement, 2),
            'weather_data': weather_data,
            'soil_data': soil_data,
            'input_conditions': {
                'soil_texture': soil_texture,
                **input_data
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'traceback': str(e.__traceback__)
        }


class IrrigationPredictor:
    # model_path = '../intelligent irrigation system/best_irrigation_model.joblib'
    
    def __init__(self, model_path = '../intelligent irrigation system/best_irrigation_model.joblib'):
        """
        Initialize the predictor with a trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model artifacts
        """
        # Load model artifacts
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.encoders = artifacts['encoders']
        self.scaler = artifacts['scaler']
        self.target_encoder = artifacts['target_encoder']
        self.features = artifacts['features']

    def prepare_input_data(self, weather_data, soil_data, crop):
        """
        Prepare input data for prediction
        
        Parameters:
        -----------
        weather_data : dict
            Dictionary containing weather-related features
        soil_data : dict
            Dictionary containing soil-related features
        crop : str
            Name of the crop
        
        Returns:
        --------
        pd.DataFrame
            Processed input data ready for prediction
        """
        # Determine season and altitude based on weather data
        season = 'wet' if weather_data['rainfall'] > 0.5 else 'dry'
        altitude = 'high' if soil_data['elevation'] > 1500 else ('mid' if soil_data['elevation'] > 800 else 'low')
        
        # Determine soil type based on soil properties
        soil_type = self._determine_soil_type(soil_data)
        
        # Create input dataframe with required features
        input_data = pd.DataFrame({
            'crop': [crop],
            'season': [season],
            'altitude': [altitude],
            'soil_type': [soil_type],
            'water_requirement_mm_day': [weather_data.get('predicted_water_requirement', 0)],
            'total_water_requirement_m3': [self._calculate_total_water_requirement(
                weather_data.get('predicted_water_requirement', 0), 
                1.0  # Default area of 1 sq km
            )]
        })
        
        # Encode categorical variables
        for column, encoder in self.encoders.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(input_data[column])
        
        # Scale numerical features
        numerical_columns = ['water_requirement_mm_day', 'total_water_requirement_m3']
        input_data[numerical_columns] = self.scaler.transform(input_data[numerical_columns])
        
        return input_data[self.features]

    def _determine_soil_type(self, soil_data):
        """Determine soil type based on soil properties"""
        # Simple logic to determine soil type based on properties
        if soil_data['water_holding_capacity'] > 0.7:
            return 'clay'
        elif soil_data['water_holding_capacity'] > 0.5:
            return 'loamy'
        elif soil_data['water_holding_capacity'] > 0.3:
            return 'silty'
        else:
            return 'sandy'

    def _calculate_total_water_requirement(self, water_requirement_mm, area_sq_km):
        """Calculate total water requirement in cubic meters"""
        water_requirement_meters = water_requirement_mm * 0.001
        area_sq_m = area_sq_km * 1e6
        return water_requirement_meters * area_sq_m

    def predict(self, weather_data, soil_data, crop):
        """
        Predict irrigation strategy
        
        Returns:
        --------
        str
            Predicted irrigation strategy
        """
        # Prepare input data
        input_data = self.prepare_input_data(weather_data, soil_data, crop)
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Decode prediction
        irrigation_strategy = self.target_encoder.inverse_transform(prediction)[0]
        
        return irrigation_strategy


        
 
def get_valid_crops():
    """
    Get list of valid crops that the model was trained on.
    Returns a list of valid crop names.
    """
    # Load the label encoder to get valid crops
    label_encoder = load('../crop recommendation/label_encoder.joblib')
    return sorted(label_encoder.classes_.tolist())

def validate_crop(crop):
    """
    Validate if the submitted crop is supported by the model.
    Returns tuple (bool, list) indicating if crop is valid and list of valid crops.
    """
    valid_crops = get_valid_crops()
    return crop.lower() in [c.lower() for c in valid_crops], valid_crops

def get_farm_prediction(location, submitted_crop):
    """
    Predict water requirement and irrigation strategy based on weather data, soil data, and submitted crop.
    """
    try:
        # Validate crop first
        is_valid_crop, valid_crops = validate_crop(submitted_crop)
        if not is_valid_crop:
            return {
                'status': 'error',
                'message': f"Invalid crop: '{submitted_crop}'. Valid crops are: {', '.join(valid_crops)}"
            }

        # Get weather data
        weather_data = get_weather_data(location)
        
        # Generate random soil data
        soil_data = generate_random_soil_data()
        
        # Predict soil texture
        latitude = weather_data['latitude']
        longitude = weather_data['longitude']
        soil_texture = predict_soil(location, latitude, longitude)
        
        # Use the validated crop (maintaining original case from valid_crops list)
        validated_crop = next(crop for crop in get_valid_crops() 
                            if crop.lower() == submitted_crop.lower())
        
        # Predict water requirement for submitted crop
        predicted_water_requirement = predict_water_requirement(
            location=location,
            weather_data=weather_data,
            soil_data=soil_data,
            crop_name=validated_crop
        )
        
        # Initialize irrigation predictor
        predictor = IrrigationPredictor()
        
        # Get irrigation strategy using submitted crop
        irrigation_strategy = predictor.predict(
            weather_data,
            soil_data,
            validated_crop
        )
        
        # Return formatted output
        return {
            'status': 'success',
            'location': location,
            'weather_data': weather_data,
            'soil_data': soil_data,
            'soil_type': soil_texture,
            'submitted_crop': validated_crop,
            'water_requirement': round(predicted_water_requirement, 2),
            'irrigation_strategy': irrigation_strategy
        }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def make_prediction(request):
    location = request.data.get('district')
    submitted_crop = request.data.get('crop')
    
    if not submitted_crop:
        return Response({
            "error": "Crop must be specified",
            "details": "Please provide a crop value in the request"
        }, status=400)
    
    print(f"\n\n Location submitted: {location}")
    print(f"Crop submitted: {submitted_crop}")

    try:

        weather_date  = get_weather_data(location)
        print(f"Weather response: {weather_date}\n")
            


        try:
            # Get prediction with submitted crop
            prediction_result = get_farm_prediction(location, submitted_crop)
            
            if prediction_result['status'] == 'success':
                # Create Prediction object
                prediction = Prediction(
                    # Metadata
                    status=prediction_result['status'],
                    location=prediction_result['location'],
                    created_by=request.user,
                    
                    # Weather data
                    temperature=prediction_result['weather_data']['temperature'],
                    humidity=prediction_result['weather_data']['humidity'],
                    wind_speed=prediction_result['weather_data']['wind_speed'],
                    rainfall=prediction_result['weather_data']['rainfall'],
                    latitude=prediction_result['weather_data']['latitude'],
                    longitude=prediction_result['weather_data']['longitude'],
                    
                    # Soil data
                    nitrogen=prediction_result['soil_data']['N'],
                    phosphorus=prediction_result['soil_data']['P'],
                    potassium=prediction_result['soil_data']['K'],
                    ph=prediction_result['soil_data']['ph'],
                    elevation=prediction_result['soil_data']['elevation'],
                    slope=prediction_result['soil_data']['slope'],
                    aspect=prediction_result['soil_data']['aspect'],
                    water_holding_capacity=prediction_result['soil_data']['water_holding_capacity'],
                    solar_radiation=prediction_result['soil_data']['solar_radiation'],
                    electrical_conductivity=prediction_result['soil_data']['ec'],
                    zinc=prediction_result['soil_data']['zn'],
                    soil_type=prediction_result['soil_type'],
                    
                    # Using submitted crop instead of predicted crop
                    predicted_crop=prediction_result['submitted_crop'],
                    water_requirement=prediction_result['water_requirement'],
                    irrigation_strategy=prediction_result['irrigation_strategy']
                )
                
                # Save to database
                prediction.save()
                
                

                
                # Serialize the saved prediction
                serializer = PredictionSerializer(prediction)
                
                print(f"Prediction data: {serializer.data}\n")
                
                # Send the password to the user's email
                send_mail(
                    subject="Your Account Password",
                    message=f"Hello, You have made an irrigation at AWUEP system. \nThe system returned the system prediction details: {serializer.data}\n\n",
                    from_email="no-reply@minagri.com",
                    recipient_list=[request.user.email],
                )
                
                return Response({
                    "message": "Prediction saved successfully",
                    "prediction": serializer.data
                }, status=201)
            else:
                return Response({
                    "error": "Prediction failed",
                    "details": prediction_result['message']
                }, status=400)
                
        except Exception as e:
            return Response({
                "error": "Failed to process prediction",
                "details": str(e)
            }, status=400)
            
    except Exception as e:
        prediction = Prediction(
            status='failed',
            location=location,
            created_by=request.user,
        )
                
        return Response({
            "error": "Location not found",
            "details": str(e)
        }, status=404)
        
        
        
        
        
# Get all predictions
@api_view(['GET'])
# @permission_classes([IsAuthenticated])
def get_all_predictions(request):
    try:
        predictions = Prediction.objects.all().order_by('-created_at')
        serializer = PredictionSerializer(predictions, many=True)
        return Response({
            "status": "success",
            "count": len(serializer.data),
            "predictions": serializer.data
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Get prediction by ID
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_prediction_by_id(request, prediction_id):
    try:
        prediction = get_object_or_404(Prediction, id=prediction_id)
        serializer = PredictionSerializer(prediction)
        return Response({
            "status": "success",
            "prediction": serializer.data
        }, status=status.HTTP_200_OK)
    except Prediction.DoesNotExist:
        return Response({
            "status": "error",
            "message": "Prediction not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Get predictions for logged-in user
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_predictions(request):
    try:
        predictions = Prediction.objects.filter(created_by=request.user).order_by('-created_at')
        serializer = PredictionSerializer(predictions, many=True)
        return Response({
            "status": "success",
            "count": len(serializer.data),
            "predictions": serializer.data
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Update prediction
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_prediction(request, prediction_id):
    try:
        prediction = get_object_or_404(Prediction, id=prediction_id)
        
        # Check if user is the owner of the prediction
        if prediction.created_by != request.user:
            return Response({
                "status": "error",
                "message": "You don't have permission to update this prediction"
            }, status=status.HTTP_403_FORBIDDEN)
        
        serializer = PredictionSerializer(prediction, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                "status": "success",
                "message": "Prediction updated successfully",
                "prediction": serializer.data
            }, status=status.HTTP_200_OK)
        return Response({
            "status": "error",
            "message": serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Delete prediction
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_prediction(request, prediction_id):
    try:
        prediction = get_object_or_404(Prediction, id=prediction_id)
        
        # Check if user is the owner of the prediction
        # if prediction.created_by != request.user:
        #     return Response({
        #         "status": "error",
        #         "message": "You don't have permission to delete this prediction"
        #     }, status=status.HTTP_403_FORBIDDEN)
        
        prediction.delete()
        return Response({
            "status": "success",
            "message": "Prediction deleted successfully"
        }, status=status.HTTP_200_OK)
    except Prediction.DoesNotExist:
        return Response({
            "status": "error",
            "message": "Prediction not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Get predictions by status
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_predictions_by_status(request, status_value):
    try:
        predictions = Prediction.objects.filter(status=status_value).order_by('-created_at')
        serializer = PredictionSerializer(predictions, many=True)
        return Response({
            "status": "success",
            "count": len(serializer.data),
            "predictions": serializer.data
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Get predictions by user phone
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_predictions_by_phone(request, phone_number):
    try:
        user = get_object_or_404(CustomUser, phone_number=phone_number)
        predictions = Prediction.objects.filter(created_by=user).order_by('-created_at')
        serializer = PredictionSerializer(predictions, many=True)
        return Response({
            "status": "success",
            "count": len(serializer.data),
            "predictions": serializer.data
        }, status=status.HTTP_200_OK)
    except CustomUser.DoesNotExist:
        return Response({
            "status": "error",
            "message": "User not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Get predictions by user email
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_predictions_by_email(request, email):
    try:
        user = get_object_or_404(CustomUser, email=email)
        predictions = Prediction.objects.filter(created_by=user).order_by('-created_at')
        serializer = PredictionSerializer(predictions, many=True)
        return Response({
            "status": "success",
            "count": len(serializer.data),
            "predictions": serializer.data
        }, status=status.HTTP_200_OK)
    except CustomUser.DoesNotExist:
        return Response({
            "status": "error",
            "message": "User not found"
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "status": "error",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
        
        


from django.db.models import Avg, Count
from django.db.models.functions import TruncMonth
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from datetime import datetime
import pandas as pd
from django.utils.timezone import now

# Water Usage Analytics
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_water_usage_analytics(request):
    try:
        # Get date range from query params or default to last 6 months
        end_date = now()
        start_date = end_date - timedelta(days=180)
        
        predictions = Prediction.objects.filter(
            created_at__range=(start_date, end_date),
            created_by=request.user
        )
        
        monthly_usage = predictions.annotate(
            month=TruncMonth('created_at')
        ).values('month').annotate(
            avg_water_requirement=Avg('water_requirement'),
            prediction_count=Count('id')
        ).order_by('month')
        
        return Response({
            "status": "success",
            "water_usage_trends": monthly_usage
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


# Soil Health Tracking
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_soil_health_metrics(request):
    try:
        predictions = Prediction.objects.filter(created_by=request.user)
        
        soil_metrics = predictions.aggregate(
            avg_nitrogen=Avg('nitrogen'),
            avg_phosphorus=Avg('phosphorus'),
            avg_potassium=Avg('potassium'),
            avg_ph=Avg('ph'),
            avg_zinc=Avg('zinc'),
            avg_ec=Avg('electrical_conductivity')
        )
        
        return Response({
            "status": "success",
            "soil_metrics": soil_metrics
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


# Crop Rotation Analysis
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analyze_crop_rotation(request):
    try:
        predictions = Prediction.objects.filter(
            created_by=request.user
        ).order_by('created_at')
        
        crop_sequence = predictions.values_list('predicted_crop', flat=True)
        unique_crops = set(crop_sequence)
        
        # Calculate crop diversity and rotation patterns
        rotation_analysis = {
            'crop_diversity': len(unique_crops),
            'crop_sequence': list(crop_sequence),
            'recommended_next_crops': _get_recommended_crops(list(crop_sequence))
        }
        
        return Response({
            "status": "success",
            "rotation_analysis": rotation_analysis
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


def _get_recommended_crops(crop_sequence):
    # Simple logic to recommend crops not recently used
    recent_crops = crop_sequence[-3:] if len(crop_sequence) > 3 else crop_sequence
    all_crops = ['Maize', 'Beans', 'Potatoes', 'Wheat', 'Sorghum', 'Soybeans']
    return [crop for crop in all_crops if crop not in recent_crops]

# Weather Impact Analysis
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analyze_weather_impact(request):
    try:
        predictions = Prediction.objects.filter(created_by=request.user)
        
        # Aggregate weather impact metrics
        weather_impact = predictions.aggregate(
            avg_temp=Avg('temperature'),
            avg_humidity=Avg('humidity'),
            avg_rainfall=Avg('rainfall'),
            avg_wind_speed=Avg('wind_speed')
        )
        
        # Convert query results to a DataFrame
        df = pd.DataFrame(list(predictions.values(
            'temperature', 'humidity', 'rainfall', 'water_requirement'
        )))
        
        # Handle missing or null values
        df = df.dropna()  # Drop rows with missing values
        if df.empty:
            return Response({
                "status": "success",
                "weather_metrics": weather_impact,
                "weather_correlations": "Not enough data to compute correlations."
            })
        
        # Calculate correlations with 'water_requirement'
        correlations = df.corr()['water_requirement'].fillna(0).to_dict()
        
        return Response({
            "status": "success",
            "weather_metrics": weather_impact,
            "weather_correlations": correlations
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
   
    

# Efficiency Metrics
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_efficiency_metrics(request):
    try:
        predictions = Prediction.objects.filter(created_by=request.user)
        
        # Calculate water use efficiency score
        efficiency_metrics = {
            'water_usage_score': _calculate_water_efficiency(predictions),
            'soil_health_score': _calculate_soil_health(predictions),
            'sustainability_score': _calculate_sustainability(predictions)
        }
        
        return Response({
            "status": "success",
            "efficiency_metrics": efficiency_metrics
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
    
    
    
    
    
def _calculate_water_efficiency(predictions):
    if not predictions:
        return 0
    avg_water_req = predictions.aggregate(Avg('water_requirement'))['water_requirement__avg']
    return min(100, max(0, 100 - (avg_water_req / 10)))




def _calculate_soil_health(predictions):
    if not predictions:
        return 0
    avg_metrics = predictions.aggregate(
        Avg('ph'),
        Avg('nitrogen'),
        Avg('phosphorus'),
        Avg('potassium')
    )
    # Simple scoring based on optimal ranges
    ph_score = 100 - abs(avg_metrics['ph__avg'] - 6.5) * 10
    return max(0, min(100, ph_score))




def _calculate_sustainability(predictions):
    if not predictions:
        return 0
    water_score = _calculate_water_efficiency(predictions)
    soil_score = _calculate_soil_health(predictions)
    return (water_score + soil_score) / 2






#admin
@api_view(['GET'])
def admin_get_water_usage_analytics(request):
    try:
        # Get date range from query params or default to last 6 months
        end_date = now()
        start_date = end_date - timedelta(days=180)
        
        predictions = Prediction.objects.filter(
            created_at__range=(start_date, end_date)
        )
        
        monthly_usage = predictions.annotate(
            month=TruncMonth('created_at')
        ).values('month').annotate(
            avg_water_requirement=Avg('water_requirement'),
            prediction_count=Count('id')
        ).order_by('month')
        
        print(f"Weather data: {get_weather_data}")
        
        return Response({
            "status": "success",
            "water_usage_trends": monthly_usage
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
    
    
    
@api_view(['GET'])
def admin_get_soil_health_metrics(request):
    try:
        predictions = Prediction.objects.all()
        
        soil_metrics = predictions.aggregate(
            avg_nitrogen=Avg('nitrogen'),
            avg_phosphorus=Avg('phosphorus'),
            avg_potassium=Avg('potassium'),
            avg_ph=Avg('ph'),
            avg_zinc=Avg('zinc'),
            avg_ec=Avg('electrical_conductivity')
        )
        
        print(f"\n\n Soil Health data: {soil_metrics}")
        
        return Response({
            "status": "success",
            "soil_metrics": soil_metrics
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
    
    
@api_view(['GET'])
def admin_analyze_crop_rotation(request):
    try:
        predictions = Prediction.objects.all(
        ).order_by('created_at')
        
        crop_sequence = predictions.values_list('predicted_crop', flat=True)
        unique_crops = set(crop_sequence)
        
        # Calculate crop diversity and rotation patterns
        rotation_analysis = {
            'crop_diversity': len(unique_crops),
            'crop_sequence': list(crop_sequence),
            'recommended_next_crops': _get_recommended_crops(list(crop_sequence))
        }
        
        print(f"\n\n Crop rotation: {rotation_analysis}\n\n")
        
        return Response({
            "status": "success",
            "rotation_analysis": rotation_analysis
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
    
    
@api_view(['GET'])
def admin_analyze_weather_impact(request):
    try:
        predictions = Prediction.objects.all()
        
        # Aggregate weather impact metrics
        weather_impact = predictions.aggregate(
            avg_temp=Avg('temperature'),
            avg_humidity=Avg('humidity'),
            avg_rainfall=Avg('rainfall'),
            avg_wind_speed=Avg('wind_speed')
        )
        
        # Convert query results to a DataFrame
        df = pd.DataFrame(list(predictions.values(
            'temperature', 'humidity', 'rainfall', 'water_requirement'
        )))
        
        # Handle missing or null values
        df = df.dropna()  # Drop rows with missing values
        if df.empty:
            return Response({
                "status": "success",
                "weather_metrics": weather_impact,
                "weather_correlations": "Not enough data to compute correlations."
            })
        
        # Calculate correlations with 'water_requirement'
        correlations = df.corr()['water_requirement'].fillna(0).to_dict()
        
        print(f"Weather impact: {weather_impact}\n")
        print(f"\n Correlations: {correlations}\n")
        
        return Response({
            "status": "success",
            "weather_metrics": weather_impact,
            "weather_correlations": correlations
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
   
    

 
@api_view(['GET'])
def admin_get_efficiency_metrics(request):
    try:
        predictions = Prediction.objects.all()
        
        # Calculate water use efficiency score
        efficiency_metrics = {
            'water_usage_score': _admin_calculate_water_efficiency(predictions),
            'soil_health_score': _admin_calculate_soil_health(predictions),
            'sustainability_score': _admin_calculate_sustainability(predictions)
        }
        
        print(f"Efficiency Metricts: {efficiency_metrics}\n\n")
        
        return Response({
            "status": "success",
            "efficiency_metrics": efficiency_metrics
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)
  



def _admin_calculate_water_efficiency(predictions):
    if not predictions:
        return 0
    avg_water_req = predictions.aggregate(Avg('water_requirement'))['water_requirement__avg']
    return min(100, max(0, 100 - (avg_water_req / 10)))




def _admin_calculate_soil_health(predictions):
    if not predictions:
        return 0
    avg_metrics = predictions.aggregate(
        Avg('ph'),
        Avg('nitrogen'),
        Avg('phosphorus'),
        Avg('potassium')
    )
    # Simple scoring based on optimal ranges
    ph_score = 100 - abs(avg_metrics['ph__avg'] - 6.5) * 10
    return max(0, min(100, ph_score))




def _admin_calculate_sustainability(predictions):
    if not predictions:
        return 0
    water_score = _calculate_water_efficiency(predictions)
    soil_score = _calculate_soil_health(predictions)
    return (water_score + soil_score) / 2







@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_get_water_usage_analytics(request):
    try:
        # Get date range from query params or default to last 6 months
        end_date = now()
        start_date = end_date - timedelta(days=180)
        
        # Filter predictions by current user
        predictions = Prediction.objects.filter(
            created_by=request.user,
            created_at__range=(start_date, end_date)
        )
        
        monthly_usage = predictions.annotate(
            month=TruncMonth('created_at')
        ).values('month').annotate(
            avg_water_requirement=Avg('water_requirement'),
            prediction_count=Count('id')
        ).order_by('month')
        
        total_predictions = predictions.count()
        avg_water_req = predictions.aggregate(Avg('water_requirement'))['water_requirement__avg'] or 0
        
        return Response({
            "status": "success",
            "water_usage_trends": monthly_usage,
            "total_predictions": total_predictions,
            "average_water_requirement": round(avg_water_req, 2)
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_get_soil_health_metrics(request):
    try:
        # Filter predictions by current user
        predictions = Prediction.objects.filter(created_by=request.user)
        
        if not predictions.exists():
            return Response({
                "status": "success",
                "message": "No predictions found for this user",
                "soil_metrics": {}
            })
        
        soil_metrics = predictions.aggregate(
            avg_nitrogen=Avg('nitrogen'),
            avg_phosphorus=Avg('phosphorus'),
            avg_potassium=Avg('potassium'),
            avg_ph=Avg('ph'),
            avg_zinc=Avg('zinc'),
            avg_ec=Avg('electrical_conductivity')
        )
        
        # Round values for better display
        for key, value in soil_metrics.items():
            if value is not None:
                soil_metrics[key] = round(value, 2)
        
        return Response({
            "status": "success",
            "soil_metrics": soil_metrics,
            "total_predictions": predictions.count()
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_analyze_crop_rotation(request):
    try:
        # Filter predictions by current user
        predictions = Prediction.objects.filter(
            created_by=request.user
        ).order_by('created_at')
        
        if not predictions.exists():
            return Response({
                "status": "success",
                "message": "No predictions found for this user",
                "rotation_analysis": {}
            })
        
        crop_sequence = list(predictions.values_list('predicted_crop', flat=True))
        unique_crops = set(crop_sequence)
        
        # Calculate crop frequency
        crop_frequency = {}
        for crop in crop_sequence:
            crop_frequency[crop] = crop_frequency.get(crop, 0) + 1
        
        # Calculate crop diversity and rotation patterns
        rotation_analysis = {
            'crop_diversity': len(unique_crops),
            'total_predictions': len(crop_sequence),
            'crop_frequency': crop_frequency,
            'most_predicted_crop': max(crop_frequency, key=crop_frequency.get) if crop_frequency else None,
            'recommended_next_crops': _get_recommended_crops(crop_sequence)
        }
        
        return Response({
            "status": "success",
            "rotation_analysis": rotation_analysis
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_analyze_weather_impact(request):
    try:
        # Filter predictions by current user
        predictions = Prediction.objects.filter(created_by=request.user)
        
        if not predictions.exists():
            return Response({
                "status": "success",
                "message": "No predictions found for this user",
                "weather_metrics": {},
                "weather_correlations": {}
            })
        
        # Aggregate weather impact metrics
        weather_impact = predictions.aggregate(
            avg_temp=Avg('temperature'),
            avg_humidity=Avg('humidity'),
            avg_rainfall=Avg('rainfall'),
            avg_wind_speed=Avg('wind_speed')
        )
        
        # Round values for better display
        for key, value in weather_impact.items():
            if value is not None:
                weather_impact[key] = round(value, 2)
        
        # Convert query results to a DataFrame
        df = pd.DataFrame(list(predictions.values(
            'temperature', 'humidity', 'rainfall', 'wind_speed', 'water_requirement'
        )))
        
        # Handle missing or null values
        df = df.dropna()
        correlations = {}
        
        if not df.empty and len(df) > 1:
            # Calculate correlations with 'water_requirement'
            correlations = df.corr()['water_requirement'].fillna(0).to_dict()
            # Remove self-correlation and round values
            correlations.pop('water_requirement', None)
            correlations = {k: round(v, 3) for k, v in correlations.items()}
        
        return Response({
            "status": "success",
            "weather_metrics": weather_impact,
            "weather_correlations": correlations,
            "total_predictions": predictions.count()
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_get_efficiency_metrics(request):
    try:
        # Filter predictions by current user
        predictions = Prediction.objects.filter(created_by=request.user)
        
        if not predictions.exists():
            return Response({
                "status": "success",
                "message": "No predictions found for this user",
                "efficiency_metrics": {}
            })
        
        # Calculate efficiency scores
        efficiency_metrics = {
            'water_usage_score': _user_calculate_water_efficiency(predictions),
            'soil_health_score': _user_calculate_soil_health(predictions),
            'sustainability_score': _user_calculate_sustainability(predictions)
        }
        
        return Response({
            "status": "success",
            "efficiency_metrics": efficiency_metrics,
            "total_predictions": predictions.count()
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_get_location_analytics(request):
    """Get analytics grouped by location for the current user"""
    try:
        predictions = Prediction.objects.filter(created_by=request.user)
        
        if not predictions.exists():
            return Response({
                "status": "success",
                "message": "No predictions found for this user",
                "location_analytics": []
            })
        
        # Group by location
        location_analytics = predictions.values('location').annotate(
            prediction_count=Count('id'),
            avg_water_requirement=Avg('water_requirement'),
            avg_temperature=Avg('temperature'),
            avg_humidity=Avg('humidity'),
            avg_ph=Avg('ph')
        ).order_by('-prediction_count')
        
        # Round values for better display
        for location in location_analytics:
            for key, value in location.items():
                if isinstance(value, float):
                    location[key] = round(value, 2)
        
        return Response({
            "status": "success",
            "location_analytics": list(location_analytics),
            "total_locations": len(location_analytics)
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_get_recent_predictions(request):
    """Get recent predictions for the current user"""
    try:
        # Get limit from query params or default to 10
        limit = int(request.GET.get('limit', 10))
        
        predictions = Prediction.objects.filter(
            created_by=request.user
        ).order_by('-created_at')[:limit]
        
        predictions_data = []
        for prediction in predictions:
            predictions_data.append({
                'id': prediction.id,
                'location': prediction.location,
                'predicted_crop': prediction.predicted_crop,
                'water_requirement': round(prediction.water_requirement, 2),
                'temperature': round(prediction.temperature, 2),
                'humidity': round(prediction.humidity, 2),
                'ph': round(prediction.ph, 2),
                'created_at': prediction.created_at,
                'irrigation_strategy': prediction.irrigation_strategy
            })
        
        return Response({
            "status": "success",
            "recent_predictions": predictions_data,
            "total_count": Prediction.objects.filter(created_by=request.user).count()
        })
    except Exception as e:
        return Response({"status": "error", "message": str(e)}, status=500)


# Helper functions for user-specific calculations
def _user_calculate_water_efficiency(predictions):
    if not predictions.exists():
        return 0
    avg_water_req = predictions.aggregate(Avg('water_requirement'))['water_requirement__avg']
    if avg_water_req is None:
        return 0
    return min(100, max(0, 100 - (avg_water_req / 10)))


def _user_calculate_soil_health(predictions):
    if not predictions.exists():
        return 0
    avg_metrics = predictions.aggregate(
        avg_ph=Avg('ph'),
        avg_nitrogen=Avg('nitrogen'),
        avg_phosphorus=Avg('phosphorus'),
        avg_potassium=Avg('potassium')
    )
    
    # Simple scoring based on optimal ranges
    ph_value = avg_metrics.get('avg_ph')
    if ph_value is None:
        return 0
    
    ph_score = 100 - abs(ph_value - 6.5) * 10
    return max(0, min(100, ph_score))


def _user_calculate_sustainability(predictions):
    if not predictions.exists():
        return 0
    water_score = _user_calculate_water_efficiency(predictions)
    soil_score = _user_calculate_soil_health(predictions)
    return round((water_score + soil_score) / 2, 2)


# You'll need to implement this function based on your crop rotation logic
def _get_recommended_crops(crop_sequence):
    """
    Implement your crop rotation recommendation logic here
    This is a placeholder implementation
    """
    if not crop_sequence:
        return []
    
    # Simple example: recommend crops that haven't been used recently
    recent_crops = set(crop_sequence[-3:]) if len(crop_sequence) >= 3 else set(crop_sequence)
    all_possible_crops = ['wheat', 'rice', 'corn', 'soybeans', 'barley', 'oats']
    
    recommended = [crop for crop in all_possible_crops if crop not in recent_crops]
    return recommended[:3]  # Return top 3 recommendations