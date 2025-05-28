from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Feedback
from .serializers import FeedbackSerializer
from irrigationApp.models import Prediction

@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def create_feedback(request):
    """ Create a new feedback entry with proper error handling """
    try:
        data = request.data.copy()
        
        # Validate required fields
        if 'prediction' not in data:
            print("Error: Missing required field 'prediction'")
            return Response(
                {"error": "Missing required field", "message": "Prediction ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        if 'rating' not in data:
            print("Error: Missing required field 'rating'")
            return Response(
                {"error": "Missing required field", "message": "Rating is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Validate and convert rate to integer
        if 'rate' in data:
            try:
                data['rate'] = int(data['rate'])
            except (ValueError, TypeError):
                print(f"Error: Invalid rate value '{data['rate']}', must be an integer")
                return Response(
                    {"error": "Invalid data", "message": "Rate must be a valid integer"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        # Validate and convert rating to integer
        try:
            data['rating'] = int(data['rating'])
            if data['rating'] < 1 or data['rating'] > 5:
                print(f"Error: Rating value {data['rating']} out of range (1-5)")
                return Response(
                    {"error": "Invalid data", "message": "Rating must be between 1 and 5"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            print(f"Error: Invalid rating value '{data['rating']}', must be an integer")
            return Response(
                {"error": "Invalid data", "message": "Rating must be a valid integer between 1 and 5"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Validate prediction exists
        try:
            prediction_id = int(data['prediction'])
            prediction = Prediction.objects.get(id=prediction_id)
            data['prediction'] = prediction_id
        except (ValueError, TypeError):
            print(f"Error: Invalid prediction ID '{data['prediction']}', must be an integer")
            return Response(
                {"error": "Invalid data", "message": "prediction ID must be a valid integer"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Prediction.DoesNotExist:
            print(f"Error: prediction with ID {data['prediction']} not found")
            return Response(
                {"error": "Not found", "message": f"prediction with ID {data['prediction']} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
        # Add the current user as creator
        data["created_by"] = request.user.id
        
        # Create feedback
        serializer = FeedbackSerializer(data=data)
        if serializer.is_valid():
            serializer.save(created_by=request.user, prediction=prediction)
            print(f"Success: Feedback created for prediction {prediction_id}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        print(f"Error: Serializer validation failed - {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    except Exception as e:
        print(f"Unexpected error during feedback creation: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def get_all_feedbacks(request):
    """ Retrieve all feedback entries """
    try:
        feedbacks = Feedback.objects.all()
        serializer = FeedbackSerializer(feedbacks, many=True)
        
        print(f"Feddbacks data: {serializer.data}\n\n")
        return Response(serializer.data)
    except Exception as e:
        print(f"Error retrieving all feedbacks: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def get_feedback_by_id(request, feedback_id):
    """ Retrieve a single feedback by ID """
    try:
        try:
            feedback_id = int(feedback_id)
        except (ValueError, TypeError):
            print(f"Error: Invalid feedback ID '{feedback_id}', must be an integer")
            return Response(
                {"error": "Invalid ID", "message": "Feedback ID must be a valid integer"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        try:
            feedback = Feedback.objects.get(id=feedback_id)
            serializer = FeedbackSerializer(feedback)
            return Response(serializer.data)
        except Feedback.DoesNotExist:
            print(f"Error: Feedback with ID {feedback_id} not found")
            return Response(
                {"error": "Not found", "message": f"Feedback with ID {feedback_id} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
    except Exception as e:
        print(f"Error retrieving feedback by ID: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["PUT"])
@permission_classes([permissions.IsAuthenticated])
def update_feedback(request, feedback_id):
    """ Update a feedback entry """
    try:
        try:
            feedback_id = int(feedback_id)
        except (ValueError, TypeError):
            print(f"Error: Invalid feedback ID '{feedback_id}', must be an integer")
            return Response(
                {"error": "Invalid ID", "message": "Feedback ID must be a valid integer"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Find the feedback and check permissions
        try:
            feedback = Feedback.objects.get(id=feedback_id)
            if feedback.created_by != request.user and not request.user.is_staff:
                print(f"Error: User {request.user.phone_number} does not have permission to update feedback {feedback_id}")
                return Response(
                    {"error": "Permission denied", "message": "You can only update your own feedback"}, 
                    status=status.HTTP_403_FORBIDDEN
                )
        except Feedback.DoesNotExist:
            print(f"Error: Feedback with ID {feedback_id} not found")
            return Response(
                {"error": "Not found", "message": f"Feedback with ID {feedback_id} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
        data = request.data.copy()
        
        # Handle rate conversion if present
        if 'rate' in data:
            try:
                data['rate'] = int(data['rate'])
            except (ValueError, TypeError):
                print(f"Error: Invalid rate value '{data['rate']}', must be an integer")
                return Response(
                    {"error": "Invalid data", "message": "Rate must be a valid integer"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        # Handle rating conversion if present
        if 'rating' in data:
            try:
                data['rating'] = int(data['rating'])
                if data['rating'] < 1 or data['rating'] > 5:
                    print(f"Error: Rating value {data['rating']} out of range (1-5)")
                    return Response(
                        {"error": "Invalid data", "message": "Rating must be between 1 and 5"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            except (ValueError, TypeError):
                print(f"Error: Invalid rating value '{data['rating']}', must be an integer")
                return Response(
                    {"error": "Invalid data", "message": "Rating must be a valid integer between 1 and 5"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        # Handle prediction ID if present
        if 'prediction' in data:
            try:
                prediction_id = int(data['prediction'])
                prediction = Prediction.objects.get(id=prediction_id)
                data['prediction'] = prediction_id
            except (ValueError, TypeError):
                print(f"Error: Invalid prediction ID '{data['prediction']}', must be an integer")
                return Response(
                    {"error": "Invalid data", "message": "prediction ID must be a valid integer"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Prediction.DoesNotExist:
                print(f"Error: prediction with ID {data['prediction']} not found")
                return Response(
                    {"error": "Not found", "message": f"prediction with ID {data['prediction']} not found"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
        
        # Update feedback
        serializer = FeedbackSerializer(feedback, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            print(f"Success: Feedback {feedback_id} updated")
            return Response(serializer.data)
            
        print(f"Error: Serializer validation failed - {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        print(f"Unexpected error during feedback update: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["DELETE"])
@permission_classes([permissions.IsAuthenticated])
def delete_feedback(request, feedback_id):
    """ Delete a feedback entry """
    try:
        try:
            feedback_id = int(feedback_id)
        except (ValueError, TypeError):
            print(f"Error: Invalid feedback ID '{feedback_id}', must be an integer")
            return Response(
                {"error": "Invalid ID", "message": "Feedback ID must be a valid integer"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Find the feedback and check permissions
        try:
            feedback = Feedback.objects.get(id=feedback_id)
            if feedback.created_by != request.user and not request.user.is_staff:
                print(f"Error: User {request.user.phone_number} does not have permission to delete feedback {feedback_id}")
                return Response(
                    {"error": "Permission denied", "message": "You can only delete your own feedback"}, 
                    status=status.HTTP_403_FORBIDDEN
                )
        except Feedback.DoesNotExist:
            print(f"Error: Feedback with ID {feedback_id} not found")
            return Response(
                {"error": "Not found", "message": f"Feedback with ID {feedback_id} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
        # Delete feedback
        feedback.delete()
        print(f"Success: Feedback {feedback_id} deleted")
        return Response({"message": "Feedback deleted successfully"}, status=status.HTTP_204_NO_CONTENT)
        
    except Exception as e:
        print(f"Unexpected error during feedback deletion: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def get_feedbacks_by_logged_in_user(request):
    """ Retrieve feedbacks created by the logged-in user """
    try:
        feedbacks = Feedback.objects.filter(created_by=request.user)
        serializer = FeedbackSerializer(feedbacks, many=True)
        print(f"Retrieved {len(feedbacks)} feedbacks for user {request.user.phone_number}")
        return Response(serializer.data)
    except Exception as e:
        print(f"Error retrieving user feedbacks: {str(e)}")
        return Response(
            {"error": "Server error", "message": f"An unexpected error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )