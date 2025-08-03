import spacy
import pandas as pd
from spacy import displacy
from tqdm import tqdm

# Load resume data
df = pd.read_csv("resume_with_topics.csv")
nlp = spacy.load("en_core_web_sm")

# Add progress bar
tqdm.pandas()

# Robust NER extraction function
def extract_named_entities(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply NER to each resume with progress bar
print("Extracting named entities...")
df["Named_Entities"] = df["Resume_str"].progress_apply(extract_named_entities)

# Save to CSV
df.to_csv("resume_with_entities.csv", index=False)
print("Named entities saved to resume_with_entities.csv")

# OPTIONAL: Visualize NER for the first resume
print("Generating NER visualization for the first resume...")

doc = nlp(df["Resume_str"][0])
html = displacy.render(doc, style="ent")
   
# Save visualization to HTML file
with open("ner_visualization.html", "w", encoding="utf-8") as f:
    f.write(html)

print("NER visualization saved to ner_visualization.html")
