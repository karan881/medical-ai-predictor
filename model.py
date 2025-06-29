# Install NLTK resources (only runs first time)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import libraries
import pandas as pd
import ast
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


# ===============================
# Load and preprocess dataset
# ===============================
df = pd.read_csv('cleaned_disease_dataset.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Convert stringified lists into Python lists
def safe_convert(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x if isinstance(x, list) else []

df['All Symptoms List'] = df['All Symptoms List'].apply(safe_convert)


# ===============================
# Load embedding model
# ===============================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# ===============================
# Text preprocessing
# ===============================
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    return lemmatizer.lemmatize(text.lower().strip())


# ===============================
# Create embeddings for symptoms
# ===============================
df['Symptom Embeddings'] = df['All Symptoms List'].apply(
    lambda sym_list: embedding_model.encode(sym_list) if sym_list else []
)


# ===============================
# Main matching function
# ===============================
def get_disease_matches(input_symptoms):
    # Preprocess input
    input_symptoms = [preprocess_text(sym) for sym in input_symptoms]

    input_words = set()
    for symptom in input_symptoms:
        input_words.update(preprocess_text(word) for word in symptom.split())

    input_embeddings = embedding_model.encode(input_symptoms)

    results = []

    for _, row in df.iterrows():
        disease_symptoms = [preprocess_text(sym) for sym in row['All Symptoms List']]

        disease_words = set()
        for sym in disease_symptoms:
            disease_words.update(preprocess_text(word) for word in sym.split())

        # Direct matches
        matched_phrases = set(input_symptoms) & set(disease_symptoms)
        matched_words = input_words & disease_words

        # Fuzzy matching
        fuzzy_matches = set()
        for input_sym in input_symptoms:
            for disease_sym in disease_symptoms:
                if fuzz.token_set_ratio(input_sym, disease_sym) > 80:
                    fuzzy_matches.add(disease_sym)

        # Embedding matching
        disease_embeddings = row['Symptom Embeddings']
        embedding_score = 0

        if len(disease_embeddings) > 0:
            cos_sim = util.cos_sim(input_embeddings, disease_embeddings)
            embedding_score = float(cos_sim.max())

        # Total score calculation
        total_score = (
            len(matched_phrases) * 1.0 +
            len(matched_words) * 0.5 +
            len(fuzzy_matches) * 0.75 +
            embedding_score * 1.2
        )

        if total_score > 0:
            combined_matches = matched_phrases.union(matched_words).union(fuzzy_matches)
            results.append({
                'Disease': row.get('Disease', 'N/A'),
                'Matched_Symptoms': sorted(combined_matches),
                'Score': round(total_score, 2),
                'Description': row.get('Description', 'N/A'),
                'Recommended_Drugs': row.get('Recommended Drugs with Dosage', 'N/A'),
                'Test_Suggestions': row.get('Test Suggestions', 'N/A'),
                'Specialist': row.get('Specialist', 'N/A')
            })

    sorted_results = sorted(results, key=lambda x: x['Score'], reverse=True)
    return sorted_results[:5]
