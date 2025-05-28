from rest_framework import serializers
from .models import Feedback

from userapp.models import CustomUser
from irrigationApp.serializers import PredictionSerializer

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'phone_number', 'email', 'role', 'created_at']

class FeedbackSerializer(serializers.ModelSerializer):
    
    created_by = UserSerializer(read_only=True)
    relocation = PredictionSerializer(read_only=True)
    
    class Meta:
        model = Feedback
        fields = "__all__"
