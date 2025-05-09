# predictor/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .forms import PredictionForm  # Import the new form

# Helper function to load pickled files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Home view (accessible to all users)
def home(request):
    return render(request, 'predictor/home.html')

# Register view
def register(request):
    try:
        if request.method == 'POST':
            form = UserCreationForm(request.POST)
            if form.is_valid():
                user = form.save()
                login(request, user)
                print(f"User {user.username} registered and logged in successfully.")
                return redirect('predictor:home')
            else:
                print("Form is invalid:", form.errors)
        else:
            form = UserCreationForm()
        return render(request, 'predictor/register.html', {'form': form})
    except Exception as e:
        print(f"Error during registration: {e}")
        return render(request, 'predictor/register.html', {'form': UserCreationForm(), 'error': str(e)})

# Login view
def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('predictor:home')
    else:
        form = AuthenticationForm()
    return render(request, 'predictor/login.html', {'form': form})

# Logout view
def user_logout(request):
    logout(request)
    return redirect('predictor:home')

# Predict view (restricted to logged-in users)
@login_required
def predict(request):
    # Load model and scaler
    model = load_pickle('predictor/static/rf_model.pkl')
    scaler = load_pickle('predictor/static/scaler.pkl')

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                size = form.cleaned_data['size']
                weight = form.cleaned_data['weight']
                sweetness = form.cleaned_data['sweetness']
                crunchiness = form.cleaned_data['crunchiness']
                juiciness = form.cleaned_data['juiciness']
                ripeness = form.cleaned_data['ripeness']
                acidity = form.cleaned_data['acidity']

                # Calculate new features
                size_sweetness = size * sweetness
                weight_size = weight / (size + 1e-6)  # Avoid division by zero

                # Prepare input data in the same order as during training
                input_data = pd.DataFrame([[size, weight, sweetness, crunchiness, juiciness, ripeness, acidity, size_sweetness, weight_size]], columns=[
                    'Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity', 'Size_Sweetness', 'Weight_Size'
                ])

                # Scale the input
                input_scaled = scaler.transform(input_data)

                # Make prediction
                prediction = model.predict(input_scaled)[0]
                result = 'Good' if prediction == 1 else 'Bad'

                return render(request, 'predictor/predict.html', {'form': form, 'result': result})
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render(request, 'predictor/predict.html', {'form': form, 'error': str(e)})
        else:
            return render(request, 'predictor/predict.html', {'form': form, 'error': 'Invalid input data. Please check your entries.'})
    else:
        form = PredictionForm()
    return render(request, 'predictor/predict.html', {'form': form})

# Dashboard view (restricted to logged-in users)
@login_required
def dashboard(request):
    try:
        # Load dataset stats
        stats = load_pickle('predictor/static/dataset_stats.pkl')
        
        # Extract stats
        num_rows = stats['num_rows']
        target_distribution = {
            'Good': stats['target_distribution'][1.0] * 100,  # Convert to percentage
            'Bad': stats['target_distribution'][0.0] * 100
        }
        feature_means = stats['feature_means']
        
        # Load model and scaler for evaluation
        model = load_pickle('predictor/static/rf_model.pkl')
        scaler = load_pickle('predictor/static/scaler.pkl')
        
        # Load dataset to compute confusion matrix
        data = pd.read_csv('predictor/static/apple_quality.csv')
        data = data.drop('A_id', axis=1)
        data['Quality'] = data['Quality'].map({'good': 1, 'bad': 0})
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Feature engineering
        data['Size_Sweetness'] = data['Size'] * data['Sweetness']
        data['Weight_Size'] = data['Weight'] / (data['Size'] + 1e-6)
        
        # Prepare features and target
        X = data[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity', 'Size_Sweetness', 'Weight_Size']]
        y = data['Quality']
        X_scaled = scaler.transform(X)
        
        # Compute accuracy and confusion matrix
        y_pred = model.predict(X_scaled)
        accuracy = (y_pred == y).mean() * 100
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred, labels=[0, 1])  # Labels: [Bad, Good]
        cm_data = cm.tolist()  # [[TN, FP], [FN, TP]]
        
        # Generate pie chart for target distribution
        plt.figure(figsize=(6, 6))
        plt.pie([target_distribution['Good'], target_distribution['Bad']], 
                labels=['Good', 'Bad'], 
                colors=['#4CAF50', '#DB4437'], 
                autopct='%1.1f%%')
        plt.title('Quality Distribution')
        pie_path = 'predictor/static/pie_chart.png'
        plt.savefig(pie_path)
        plt.close()
        
        # Feature importance plot (already saved in notebook)
        feature_importance_path = 'predictor/static/feature_importance.png'
        
        return render(request, 'predictor/dashboard.html', {
            'num_rows': num_rows,
            'target_distribution': target_distribution,
            'feature_means': feature_means,
            'accuracy': round(accuracy, 2),
            'confusion_matrix_data': cm_data,
            'pie_chart': '/static/pie_chart.png',
            'feature_importance': '/static/feature_importance.png'
        })
    except Exception as e:
        print(f"Error in dashboard view: {e}")
        return render(request, 'predictor/dashboard.html', {'error': str(e)})