"""Example of using PandasAI with a CSV file and Google Vertexai."""

import os

import pandas as pd

from pandasai import Agent
from pandasai.llm import GoogleGemini

df = pd.read_csv("examples/data/Loan payments data.csv")

# Set the path of your json credentials
llm = GoogleGemini(api_key=os.getenv("GEMINI_API_KEY"))
agent = Agent(df, config={"llm": llm})
response = agent.chat("How many loans are from men and have been paid off?")
print(response)
# Output: 247 loans have been paid off by men.
