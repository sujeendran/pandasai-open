"""Example of using PandasAI with a CSV file and Google Vertexai."""

import os

import pandas as pd

from pandasai import Agent
from pandasai.llm import GoogleGemini

df = pd.read_csv("examples/data/Loan payments data.csv")

# Set the path of your json credentials
llm = GoogleGemini(api_key=os.getenv("GEMINI_API_KEY"))
agent = Agent(df, config={"llm": llm, "data_viz_library": "plotly", "enable_cache": False})
response = agent.chat("How many loans are from men and have been paid off?")
print(response)
# Output: 247 loans have been paid off by men.

# Get Clarification Questions
questions = agent.clarification_questions("Who paid the most?")
for question in questions:
    print(question)

# Explain how the chat response is generated
response = agent.explain()
print(response)

# Plot a chart
response = agent.chat("Plot a bar chart showing the count of unpaid loans by gender")
print(response)