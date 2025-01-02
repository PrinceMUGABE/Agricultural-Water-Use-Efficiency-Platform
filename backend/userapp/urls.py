from django.urls import path
from .views import (
    register_user,
    login_user,
    reset_password,
    list_all_users,
    get_user_by_email,
    get_user_by_id,
    get_user_by_phone,
    update_user,
    delete_user_by_id,
    contact_us,

)

urlpatterns = [
    # User Registration
    path('register/', register_user, name='register_user'),
    
    # User Login
    path('login/', login_user, name='login_user'),
    
    # Reset Password
    path('forget_password/', reset_password, name='reset_password'),
    
    # User Management
    path('users/', list_all_users, name='list_all_users'),  # List all users (admin only)
    path('user/<int:user_id>/', get_user_by_id, name='get_user_by_id'),  # Get a user by ID
    path('update/<int:user_id>/', update_user, name='update_user'),  # Update a user
    path('delete/<int:user_id>/', delete_user_by_id, name='delete_user_by_id'),  # Delete a user by ID
    path('email/', get_user_by_email, name='get_user_by_email'),  # Get a user by email
    path('phone/', get_user_by_phone, name='get_user_by_phone'),  # Get a user by phone number
    path('contact/', contact_us, name='contact'),

]