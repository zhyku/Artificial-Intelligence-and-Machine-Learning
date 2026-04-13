import streamlit as st
from apikey import apikey

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper

import os

#CONFIG

os.environ["OPENAI_API_KEY"] = apikey

st.set_page_config(page_title="YouTube GPT Creator")

#LLM + TOOLS

llm = OpenAI(temperature=0.9)
wiki = WikipediaAPIWrapper()

title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a YouTube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="Write a YouTube video script based on this title: {title}. Use this Wikipedia research: {wikipedia_research}"
)

title_chain = title_template | llm
script_chain = script_template | llm

#HELPERS (production safety)
def validate_input(text: str) -> bool:
    return text is not None and len(text.strip()) > 0


def generate_content(topic: str):
    title = title_chain.invoke({"topic": topic})

    wiki_data = ""
    try:
        wiki_data = wiki.run(topic)
    except Exception:
        wiki_data = "Wikipedia data not available."

    script = script_chain.invoke({
        "title": title,
        "wikipedia_research": wiki_data
    })

    return title, script, wiki_data

#UI
st.title("YouTube GPT Creator")

prompt = st.text_input("Enter your topic")

if prompt:
    if not validate_input(prompt):
        st.error("Please enter a valid topic.")
    else:
        try:
            with st.spinner("Generating content..."):
                title, script, wiki_data = generate_content(prompt)

            st.subheader("Title")
            st.write(title)

            st.subheader("Script")
            st.write(script)

            with st.expander("Wikipedia Research"):
                st.info(wiki_data)

        except Exception as e:
            st.error(f"Error: {e}")