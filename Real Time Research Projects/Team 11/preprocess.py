import pandas as pd
import spacy
import re
from tqdm import tqdm

# Enable tqdm progress bar for pandas
tqdm.pandas()

# 1. Load spaCy English model
# Make sure you have run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# 2. Define your text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+\.[a-z]+", "", text)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # spaCy processing
    doc = nlp(text)
    # Lemmatize, remove stopwords, short tokens, and non-alpha tokens
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]
    return " ".join(tokens)

# 3. Load your original resume CSV
# Make sure the path and column name are correct!
df = pd.read_csv(r"C:\Users\G . SANJAY\OneDrive\Desktop\resume analyser\Resume.csv")

# 4. Apply preprocessing to the resume text column
# Replace 'Resume_str' with the actual column name if different
df["Clean_Resume"] = df["Resume_str"].progress_apply(preprocess_text)

# 5. Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_resume.csv", index=False)
print("Cleaned resumes saved to cleaned_resume.csv.")

# 6. Example: Load the cleaned file to verify
df_cleaned = pd.read_csv("cleaned_resume.csv")
print(df_cleaned.head())
