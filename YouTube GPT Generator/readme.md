YouTube GPT Creator

A simple AI-powered Streamlit app that generates YouTube video titles and scripts using OpenAI + Wikipedia context.

Features:
- Generate YouTube video titles from any topic
- Create full video scripts using AI
- Uses Wikipedia for extra context
- Simple web interface with Streamlit

Tech Stack:
- Python
- Streamlit
- LangChain
- OpenAI API
- Wikipedia API

Installation:

Clone the repo:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies: pip install -r requirements.txt

API Key Setup: This project requires an OpenAI API key.

Get your key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Local setup:

Create a file named `apikey.py`:

apikey = "your-openai-api-key"

Streamlit Cloud setup: Add this in Secrets - OPENAI_API_KEY = your-openai-api-key

Run the App: streamlit run app.py

Notes:

* Do NOT share your API key publicly
* Wikipedia is used for additional context in script generation
* This project is for learning and demo purposes

Output:

* YouTube video title
* Full video script
* Wikipedia research context

Free to use for personal and educational projects
