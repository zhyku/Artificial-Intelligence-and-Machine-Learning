from flask import Flask, render_template, request, redirect, url_for
import os
import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Sample skills database
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
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_skills(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    found_skills = [skill for skill in skills_db if skill in filtered]
    return list(set(found_skills))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return "No file part"
    file = request.files['resume']
    if file.filename == '':
        return "No selected file"

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract and process resume
    text = extract_text_from_pdf(filepath)
    skills = extract_skills(text)

    return render_template('index.html', filename=filename, skills=skills, resume_text=text)
if __name__ == '__main__':
    app.run(debug=True)
