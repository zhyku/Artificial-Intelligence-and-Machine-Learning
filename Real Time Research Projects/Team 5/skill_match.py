# skill_match.py

import pandas as pd

def calculate_match_score(input_resume_skills, required_skills):
    """
    Calculates percentage match between extracted skills from resume and required skills for the job.
    """
    req_skills = [skill.strip().lower() for skill in required_skills.split(",")]
    match_scores = []

    for skills in input_resume_skills:
        matched = sum(1 for skill in req_skills if skill in skills)
        score = round((matched / len(req_skills)) * 100, 2)
        match_scores.append(score)
    
    return match_scores

# Load the resumes with extracted skills
df = pd.read_csv("resume_with_skills.csv")

# Define required skills (example)
required_skills = "Python, Machine Learning, Deep Learning, SQL, TensorFlow, PyTorch, Scikit-Learn, NLP, Git, GitHub"

# Convert the 'Extracted_Skills' string back to a list
df["Extracted_Skills"] = df["Extracted_Skills"].apply(eval)

# Calculate match scores
df["Match_Score"] = calculate_match_score(df["Extracted_Skills"], required_skills)

# Save results
df.to_csv("resume_with_scores.csv", index=False)
print("Match scores saved to resume_with_scores.csv")
