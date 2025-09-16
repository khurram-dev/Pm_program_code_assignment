# -*- coding: utf-8 -*-

# !pip install --upgrade langgraph langchain-google-genai wikipedia

# Multi-agent with Supervisor Node

# Resource: https://github.com/AiAgentGuy/LangGraph-supervisor/blob/main/src/supervisor_graph.py

# This is my practice AI program.
# I am still learning AI programming and my understanding is not much, but I tried to build this project.

# ## What it does
# - The program has a **Coordinator Agent**.
# - The coordinator takes the user question and sends it to the **Research Node**.
# - The research node searches Wikipedia for the question.
# - The results are then passed to the **Analyze Node**.
# - The analyze node creates a simple summary from the Wikipedia content.

# ## Why I made this
# I wanted to practice using AI agents step by step.
# This project is very basic, but it helped me understand how an agent can search and then summarize information.

from __future__ import annotations

from typing import Annotated, Literal, TypedDict
import wikipedia
from pprint import pprint

from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.graph.message import add_messages

# GOOGLE_API_KEY
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
# GROQ_API_KEY=userdata.get('GROQ_API_KEY')

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # You can change the Gemini model here
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Graph state
class State(TypedDict):
  messages: Annotated[list, add_messages]
  current_stage: str
  completed_stages: list[str]

# Research node, i am using wikipedia for searching
def research_agent(query):
  """Research agent, it analyses the question and search from internet and send response back to coordinator_node"""

  # Wikipedia search the query, and reponse all page content
  results = wikipedia.search(query, results=10)
  pages = []

  for title in results:
    try:
      page = wikipedia.page(title, auto_suggest=False)
      pages.append({
        "title": page.title,
        "url": page.url,
        "content": page.content
      })
    except Exception:
      continue  # skip errors, only get result pages, 10 is enough to analyze and generate result of query

  return pages

# Pages Result Analyze node, it will analyze all results and create summary of user's query
def analyze_wikipedia_results(state: State, pages):
  """LLM based analyze_wikipedia_results node, it will check all pages content and then create a good summary."""

  # Combine all Wikipedia page contents into one string
  pages_text = "\n\n".join([f"### {p['content']}" for p in pages if "content" in p])

  # System prompt
  system_prompt = f"""You are an analysis agent. Your role is to carefully analyze the response received as pages.
  - - Always keep the original user question in mind when analyzing.
  - Summarize the key points clearly and concisely.
  - Highlight only the most important information relevant to the query.
  - If the content is too long, condense it without losing the meaning.
  - Do not add unrelated commentary or personal opinions.
  - Output should be structured, factual, and easy to understand.
  Question: {state["messages"][-1]["content"]}
  """

  # Construct messages
  llm_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Here are the Wikipedia pages to analyze:\n\n{pages_text}"}
  ]

  # print(llm_prompt)

  llm_response = llm_model.invoke(llm_prompt)

  # print(llm_response.content)

  return {"messages": [llm_response.content]}

# analyze_wikipedia_results({"messages": [
#     {
#       'role': 'user',
#       'content': 'Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.'
#     }
# ]}, ['test1', 'test2', 'test3'])

# LLM based coordinator node, handles all sub-nodes requests & reponses
def coordinator_node(state: State):
  """LLM based coordinator node, it acts like a supervisor and handles all other nodes requests & reponses."""

  # Update Graph state
  completed = state.get("completed_stages", [])
  current = state.get("current_stage", "")

  # Get user query
  query = state["messages"][-1]["content"]

  # Check the query and if it is not in memory, then call search node, get question content
  # for now memory is False, will add code later
  memory_check = False

  if not memory_check:
    # Get query content
    wpedia_results = research_agent(query)

    if wpedia_results:
        # Analyze the result and generate summary
        analyses_result = analyze_wikipedia_results({"messages": [
          {
            'role': 'user',
            'content': 'Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.'
          }
        ]}, wpedia_results)

        print(analyses_result["messages"])

    else:
      print("No results found.")

coordinator_node({"messages": [
    {
      'role': 'user',
      'content': 'Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.'
    }
]})
