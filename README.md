# HI-744 Programming Assignment

This project compares patient case reports using TF-IDF and BM25. It ranks the five most similar reports and evaluates the results.

## Requirements

Install the required Python packages:

    py -m pip install -r requirements.txt

Download the NLTK language resources:

    py -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

## Run

Pass the path to a folder containing the input JSON files:

    py main.py "C:\path\to\your\patient_data"

The patient data is not included in this repository.
