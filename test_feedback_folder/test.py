import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict 
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

# Import the prompts from prompt.py
import prompts

# Initialize LLMs and embeddings
def initialize_llms():
    llm_params = {
        "model": "gpt-3.5-turbo", 
        "temperature": 0.4
    }
    
    metric_llms = {
        "empathy_score": ChatOpenAI(**llm_params),
        "clarity_and_conciseness": ChatOpenAI(**llm_params),
        "grammar_and_language": ChatOpenAI(**llm_params),
        "listening_score": ChatOpenAI(**llm_params),
        "problem_resolution_effectiveness": ChatOpenAI(**llm_params),
        "personalisation_index": ChatOpenAI(**llm_params),
        "conflict_management": ChatOpenAI(**llm_params),
        "response_time": ChatOpenAI(**llm_params),
        "customer_satisfiction_score": ChatOpenAI(**llm_params),
        "positive_sentiment_score": ChatOpenAI(**llm_params),
        "structure_and_flow": ChatOpenAI(**llm_params),
        "stuttering_words": ChatOpenAI(**llm_params),
        "active_listening_skills": ChatOpenAI(**llm_params),
        "rapport_building": ChatOpenAI(**llm_params),
        "engagement": ChatOpenAI(**llm_params),
        "feedback": ChatOpenAI(**llm_params)
    }
    
    return metric_llms

# Define the GraphState with necessary fields
class GraphState(TypedDict):
    transcript: str
    empathy_score: str
    clarity_and_conciseness: str
    grammar_and_language: str 
    listening_score: str
    problem_resolution_effectiveness: str
    personalisation_index: str
    conflict_management: str 
    response_time: str
    customer_satisfiction_score: str
    positive_sentiment_score: str
    structure_and_flow: str
    stuttering_words: str
    active_listening_skills: str
    rapport_building: str
    engagement: str
    feedback: str

# Generic metric calculation function
def calculate_metric(state, metric_name, prompt_func, llm):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompt_func(transcript))
    chain = prompt | llm | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state[metric_name] = score
    return state

# Create node functions dynamically
def create_metric_node(metric_name, prompt_func, llm):
    def node(state):
        return calculate_metric(state, metric_name, prompt_func, llm)
    return node

# Set up the workflow with parallel execution
def create_parallel_workflow():
    # Initialize LLMs
    metric_llms = initialize_llms()
    
    # Define metric configurations
    metric_configs = [
        ("empathy_score", prompts.empathy_score_prompt),
        ("clarity_and_conciseness", prompts.clarity_and_conciseness_prompt),
        ("grammar_and_language", prompts.grammar_and_language_prompt),
        ("listening_score", prompts.listening_score_prompt),
        ("problem_resolution_effectiveness", prompts.problem_resolution_effectiveness_prompt),
        ("personalisation_index", prompts.personalisation_index_prompt),
        ("conflict_management", prompts.conflict_management_prompt),
        ("response_time", prompts.response_time_prompt),
        ("customer_satisfiction_score", prompts.customer_satisfiction_score_prompt),
        ("positive_sentiment_score", prompts.positive_sentiment_score_prompt),
        ("structure_and_flow", prompts.structure_and_flow_prompt),
        ("stuttering_words", prompts.stuttering_words_prompt),
        ("active_listening_skills", prompts.active_listening_skills_prompt),
        ("rapport_building", prompts.rapport_building_prompt),
        ("engagement", prompts.engagement_prompt),
        ("feedback", prompts.feedback_prompt)
    ]
    
    # Create workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes dynamically
    for metric_name, prompt_func in metric_configs:
        node = create_metric_node(metric_name, prompt_func, metric_llms[metric_name])
        workflow.add_node(f"node_{metric_name}", node)
    
    # Add an aggregation/end node
    def aggregate_results(state):
        return state
    
    workflow.add_node("end_node", aggregate_results)
    
    # Connect all nodes to the end node
    for metric_name, _ in metric_configs:
        workflow.add_edge(f"node_{metric_name}", "end_node")
    
    # Set the entry and end points
    workflow.set_entry_point("node_empathy_score")
    workflow.add_edge("end_node", END)
    
    # Compile the graph
    return workflow.compile()

# Usage
def analyze_customer_interaction(transcript):
    # Create the workflow
    customer_graph = create_parallel_workflow()
    
    # Initialize the state
    initial_state = {
        "transcript": transcript
    }
    
    # Execute the graph
    result = customer_graph.invoke(initial_state)
    
    return result

# Optional: Generate final scorecard
def generate_final_scorecard(result):
    scores = {
        key: value for key, value in result.items() 
        if key != 'transcript' and isinstance(value, str)
    }
    return scores


# Example usage
transcript = "Your customer interaction transcript here"
result = analyze_customer_interaction(transcript)

# Optional: Generate a scorecard
scorecard = generate_final_scorecard(result)
print(scorecard)