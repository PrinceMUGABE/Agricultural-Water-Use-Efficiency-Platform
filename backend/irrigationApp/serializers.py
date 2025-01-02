from rest_framework import serializers
from userapp.models import CustomUser
from .models import Prediction



class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'phone_number', 'email', 'role', 'created_at']

class PredictionSerializer(serializers.ModelSerializer):
    created_by = UserSerializer()  # Nest the UserSerializer to include user information

    class Meta:
        model = Prediction
        fields = [
            'id', 'status', 'location',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'latitude', 'longitude', 'nitrogen', 'phosphorus',
            'potassium', 'ph', 'elevation', 'slope', 'aspect',
            'water_holding_capacity', 'solar_radiation',
            'electrical_conductivity', 'zinc', 'soil_type',
            'predicted_crop', 'water_requirement', 'irrigation_strategy',
            'created_by', 'created_at'
        ]
