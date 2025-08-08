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
    # Programming Languages & Frameworks
    'python', 'java', 'c++', 'c', 'c#', 'javascript', 'typescript', 'html', 'css', 'php', 'ruby', 'go', 'swift', 'kotlin',
    'r', 'scala', 'perl', 'matlab', 'bash', 'shell scripting', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
    'fastapi', 'flask', 'django', 'spring', 'react', 'angular', 'vue.js', 'node.js', 'express.js', 'dotnet', '.net', 'asp.net',

    # Data Science & Analytics
    'machine learning', 'deep learning', 'data science', 'data analysis', 'data mining', 'data visualization',
    'nlp', 'natural language processing', 'computer vision', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv',
    'scikit-learn', 'pytorch', 'tensorflow', 'keras', 'big data', 'hadoop', 'spark', 'power bi', 'tableau', 'excel',

    # Cloud & DevOps
    'aws', 'azure', 'google cloud', 'gcp', 'cloud computing', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform',
    'ansible', 'linux', 'unix', 'windows server', 'devops', 'cloudformation', 'serverless', 'microservices',

    # Software & Tools
    'git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'trello', 'microsoft office', 'powerpoint', 'word',
    'outlook', 'notion', 'asana', 'zoom', 'skype', 'teams', 'vs code', 'eclipse', 'intellij', 'android studio',

    # Business, Management & Soft Skills
    'project management', 'agile', 'scrum', 'kanban', 'waterfall', 'product management', 'business analysis',
    'stakeholder management', 'risk management', 'change management', 'operations management', 'crm', 'erp',
    'leadership', 'teamwork', 'collaboration', 'communication', 'presentation', 'public speaking', 'negotiation',
    'problem-solving', 'critical thinking', 'decision making', 'adaptability', 'creativity', 'time management',
    'organizational skills', 'attention to detail', 'multitasking', 'conflict resolution', 'emotional intelligence',

    # HR & Administration
    'hr', 'recruitment', 'talent acquisition', 'employee relations', 'training', 'onboarding', 'performance management',
    'feedback', 'supervision', 'record keeping', 'inquiry management', 'payroll', 'benefits administration',
    'compliance', 'labor laws',

    # Marketing & Sales
    'marketing', 'digital marketing', 'seo', 'sem', 'content creation', 'copywriting', 'social media', 'email marketing',
    'google analytics', 'market research', 'branding', 'sales', 'lead generation', 'customer relationship management',

    # Customer Service & Support
    'customer service', 'customer support', 'complaint resolution', 'inquiry management', 'call center', 'crm',

    # Finance & Accounting
    'accounting', 'bookkeeping', 'financial analysis', 'budgeting', 'forecasting', 'taxation', 'auditing', 'quickbooks',
    'sap', 'oracle',

    # Design & Creative
    'graphic design', 'ui design', 'ux design', 'adobe photoshop', 'adobe illustrator', 'adobe xd', 'figma', 'sketch',
    'canva', 'video editing', 'animation', '3d modeling',

    # Miscellaneous / In-demand
    'blockchain', 'cybersecurity', 'penetration testing', 'network security', 'iot', 'robotics', 'virtual reality',
    'augmented reality', 'supply chain', 'logistics', 'quality assurance', 'qa', 'testing', 'sustainability',
    'foreign languages', 'spanish', 'french', 'german', 'mandarin', 'japanese', 'arabic'
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
