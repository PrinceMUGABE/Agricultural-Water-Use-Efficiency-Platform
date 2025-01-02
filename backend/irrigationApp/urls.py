
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('predict/', views.make_prediction, name='predict'),
    

    path('predictions/', views.get_all_predictions, name='get_all_predictions'),
    path('predictions/<int:prediction_id>/', views.get_prediction_by_id, name='get_prediction_by_id'),
    path('my-predictions/', views.get_user_predictions, name='get_user_predictions'),
    path('predictions/<int:prediction_id>/update/', views.update_prediction, name='update_prediction'),
    path('prediction/delete/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    path('predictions/status/<str:status_value>/', views.get_predictions_by_status, name='get_predictions_by_status'),
    path('predictions/phone/<str:phone_number>/', views.get_predictions_by_phone, name='get_predictions_by_phone'),
    path('predictions/email/<str:email>/', views.get_predictions_by_email, name='get_predictions_by_email'),
    
    
    path('analytics/water-usage/', views.get_water_usage_analytics, name='water_usage_analytics'),
    path('analytics/soil-health/', views.get_soil_health_metrics, name='soil_health_metrics'),
    path('analytics/crop-rotation/', views.analyze_crop_rotation, name='crop_rotation_analysis'),
    path('analytics/weather-impact/', views.analyze_weather_impact, name='weather_impact_analysis'),
    path('analytics/efficiency/', views.get_efficiency_metrics, name='efficiency_metrics'),
    
    
    
    path('analytics/water-usage/admin/', views.admin_get_water_usage_analytics, name='water_usage_analytics'),
    path('analytics/soil-health/admin/', views.admin_get_soil_health_metrics, name='soil_health_metrics'),
    path('analytics/crop-rotation/admin/', views.admin_analyze_crop_rotation, name='crop_rotation_analysis'),
    path('analytics/weather-impact/admin/', views.admin_analyze_weather_impact, name='weather_impact_analysis'),
    path('analytics/efficiency/admin/', views.admin_get_efficiency_metrics, name='efficiency_metrics'),

]


























