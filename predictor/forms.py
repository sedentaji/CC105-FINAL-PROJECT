# predictor/forms.py
from django import forms

class PredictionForm(forms.Form):
    size = forms.FloatField(label="Size", required=True)
    weight = forms.FloatField(label="Weight", required=True)
    sweetness = forms.FloatField(label="Sweetness", required=True)
    crunchiness = forms.FloatField(label="Crunchiness", required=True)
    juiciness = forms.FloatField(label="Juiciness", required=True)
    ripeness = forms.FloatField(label="Ripeness", required=True)
    acidity = forms.FloatField(label="Acidity", required=True)