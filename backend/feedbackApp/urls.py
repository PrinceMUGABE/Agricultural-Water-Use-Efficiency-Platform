from django.urls import path
from .views import (
    create_feedback,
    get_all_feedbacks,
    get_feedback_by_id,
    update_feedback,
    delete_feedback,
    get_feedbacks_by_logged_in_user
)

urlpatterns = [
    path("feedbacks/", get_all_feedbacks, name="get_all_feedbacks"),
    path("create/", create_feedback, name="create_feedback"),
    path("<int:feedback_id>/", get_feedback_by_id, name="get_feedback_by_id"),
    path("update/<int:feedback_id>/", update_feedback, name="update_feedback"),
    path("delete/<int:feedback_id>/", delete_feedback, name="delete_feedback"),
    path("my-feedbacks/", get_feedbacks_by_logged_in_user, name="get_feedbacks_by_logged_in_user"),
]
