from django.db import models
from django.conf import settings

class Prediction(models.Model):
    # Prediction Metadata
    status = models.CharField(max_length=50)
    location = models.CharField(max_length=100)
    
    # Weather Data
    temperature = models.FloatField()
    humidity = models.FloatField()
    wind_speed = models.FloatField()
    rainfall = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    
    # Soil Data
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    ph = models.FloatField()
    elevation = models.FloatField()
    slope = models.FloatField()
    aspect = models.FloatField()
    water_holding_capacity = models.FloatField()
    solar_radiation = models.FloatField()
    electrical_conductivity = models.FloatField()
    zinc = models.FloatField()
    soil_type = models.CharField(max_length=100)
    
    # Prediction Results
    predicted_crop = models.CharField(max_length=100)
    water_requirement = models.FloatField()
    irrigation_strategy = models.CharField(max_length=200)

    # Metadata
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.location} - {self.predicted_crop}"
