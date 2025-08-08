import pandas as pd
import re

# === STEP 1: Load Your Preprocessed Dataset === #
# Make sure 'cleaned_resume.csv' exists and has 'Clean_Resume' and 'Resume_str' columns
df = pd.read_csv("cleaned_resume.csv")

# Fill NaN values in 'Clean_Resume' with empty strings to avoid errors
df['Clean_Resume'] = df['Clean_Resume'].fillna("")

# === STEP 2: Define Skills List === #
skills_db = [
    'python', 'sql', 'excel', 'c++', 'java', 'machine learning', 'deep learning',
    'data analysis', 'nlp', 'pytorch', 'tensorflow', 'hr', 'communication',
    'customer service', 'project management', 'git', 'github', 'data visualization',
    'marketing', 'training', 'recruitment', 'employee relations', 'oop',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv', 'fastapi', 'flask',
    'power bi', 'tableau', 'linux', 'html', 'css', 'javascript', 'leadership''teamwork',
    'customer support', 'complaint resolution', 'customer service', 'leadership',
    'problem-solving', 'team lead', 'organizational skills', 'record keeping', 'supervision',
    'feedback', 'performance management', 'inquiry management'

]

# === STEP 3: Define Extraction Function === #
def extract_skills(text, skills_db):
    # Ensure input is a string
    if not isinstance(text, str):
        return []
    text = text.lower()
    extracted = []
    for skill in skills_db:
        # Use word boundaries for multi-word skills to avoid partial matches
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            extracted.append(skill)
    return extracted

# === STEP 4: Apply Extraction to DataFrame === #
df['Extracted_Skills'] = df['Clean_Resume'].apply(lambda x: extract_skills(x, skills_db))

# === STEP 5: Save or View Results === #
print(df[['Resume_str', 'Extracted_Skills']].head(5))

# (Optional) Save to file
df.to_csv("resume_with_skills.csv", index=False)
print("Results saved to resume_with_skills.csv")
