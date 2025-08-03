# topic_modeling.py

import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import nltk
from nltk.corpus import stopwords
import re
import warnings

warnings.filterwarnings("ignore")

# Ensure stopwords are available
nltk.download('stopwords')
print("Stopwords downloaded, continuing script...")

# Load preprocessed resumes
df = pd.read_csv("resume_with_scores.csv")
print("Loaded resume_with_scores.csv.")

# Fill NaN values in Clean_Resume with empty strings
df["Clean_Resume"] = df["Clean_Resume"].fillna("")

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove punctuation
    tokens = text.lower().split()  # tokenize and lowercase
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

print("Preprocessing text...")
# Apply preprocessing
docs = df["Clean_Resume"].apply(preprocess_text)
print("Text preprocessing complete.")

# Create dictionary and BoW
print("Building dictionary and Bag-of-Words corpus...")
dictionary = corpora.Dictionary(docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
print("Dictionary and corpus built.")

# Build LDA model
num_topics = 4
print(f"Training LDA model with {num_topics} topics...")
lda_model = gensim.models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=10,  # Reduce passes for faster runs; increase for better results
    random_state=42,
    minimum_probability=0.01
)
print("LDA model training complete.")

# Print topics
print("\nDiscovered Topics:")
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic #{idx + 1}: {topic}")

# Save topic distribution to dataframe
print("Calculating topic distributions for each resume...")
topic_dist = [lda_model[doc] for doc in bow_corpus]
df["Topic_Distribution"] = topic_dist

# Save output
df.to_csv("resume_with_topics.csv", index=False)
print("Topic modeling results saved to resume_with_topics.csv")

# OPTIONAL: Visualize
print("Preparing LDA visualization (this may take a while)...")
vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis, "lda_visualization.html")
print("LDA visualization saved to lda_visualization.html")
