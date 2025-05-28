from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime
from userapp.models import CustomUser
from irrigationApp.models import Prediction


class Feedback(models.Model):
    created_by = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name="feedback", null=True, blank=True)
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, null=True, blank=True)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)], help_text="Rating from 1 to 5")
    comment = models.TextField()
    rate = models.IntegerField(default=0)  # Added default value of 0
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback by {self.created_by.phone_number} - Rating: {self.rating}"