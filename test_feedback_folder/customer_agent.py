# main.py

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
# from prompts import empathy_score_prompt, clarity_and_conciseness_prompt, grammar_and_language_prompt, listening_score_prompt, problem_resolution_effectiveness_prompt,personalisation_index_prompt,conflict_management_prompt,response_time_prompt,customer_satisfiction_score_prompt,positive_sentiment_score_prompt,structure_and_flow_prompt,stuttering_words_prompt,product_knowledge_score_prompt,persuasion_and_negotiation_skills_prompt,objection_handling_prompt,confidence_score_prompt,value_proposition_prompt,pitch_quality_prompt,call_to_action_effectiveness_prompt,questioning_technique_prompt,rapport_building_prompt,active_listening_skills_prompt,upselling_success_rate_prompt,engagement_prompt,stuttering_words_prompt
import prompts

# Initialize LLMs and embeddings

# Customer service metrics
llm_empathy_score = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_clarity_and_conciseness = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_grammar_and_language = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_listening_score = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_problem_resolution_effectiveness = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_personalisation_index = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_conflict_management = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_response_time = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_customer_satisfiction_score = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_positive_sentiment_score = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_structure_and_flow = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_stuttering_words = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_active_listening_skills = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_rapport_building = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_engagement = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)
llm_feedback = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0.4)

embeddings = OpenAIEmbeddings()




# Define the GraphState with necessary fields
from typing_extensions import TypedDict
class GraphState(TypedDict):
    transcript: str

    empathy_score :str
    clarity_and_conciseness : str
    grammar_and_language : str 
    listening_score :str
    problem_resolution_effectiveness : str
    personalisation_index : str
    conflict_management : str 
    response_time : str
    customer_satisfiction_score : str
    positive_sentiment_score : str
    structure_and_flow : str
    stuttering_words : str
    active_listening_skills : str
    rapport_building : str
    engagement : str
    feedback : str


# Node functions for each metric
def empathy_score(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.empathy_score_prompt(transcript))
    chain = prompt | llm_empathy_score | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["empathy_score"] = score
    return state

def clarity_and_conciseness(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.clarity_and_conciseness_prompt(transcript))
    chain = prompt | llm_clarity_and_conciseness | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["clarity_and_conciseness"] = score
    return state

def grammar_and_language(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.grammar_and_language_prompt(transcript))
    chain = prompt | llm_grammar_and_language | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["grammar_and_language"] = score
    return state

def listening_score(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.listening_score_prompt(transcript))
    chain = prompt | llm_listening_score | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["listening_score"] = score
    return state

def problem_resolution_effectiveness(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.problem_resolution_effectiveness_prompt(transcript))
    chain = prompt | llm_problem_resolution_effectiveness | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["problem_resolution_effectiveness"] = score
    return state

def personalisation_index(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.personalisation_index_prompt(transcript))
    chain = prompt | llm_personalisation_index | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["personalisation_index"] = score
    return state

def conflict_management(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.conflict_management_prompt(transcript))
    chain = prompt | llm_conflict_management | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["conflict_management"] = score
    return state

def response_time(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.response_time_prompt(transcript))
    chain = prompt | llm_response_time | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["response_time"] = score
    return state

def customer_satisfiction_score(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.customer_satisfiction_score_prompt(transcript))
    chain = prompt | llm_customer_satisfiction_score | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["customer_satisfiction_score"] = score
    return state

def positive_sentiment_score(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.positive_sentiment_score_prompt(transcript))
    chain = prompt | llm_positive_sentiment_score | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["positive_sentiment_score"] = score
    return state

def structure_and_flow(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.structure_and_flow_prompt(transcript))
    chain = prompt | llm_structure_and_flow | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["structure_and_flow"] = score
    return state

def active_listening_skills(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.active_listening_skills_prompt(transcript))
    chain = prompt | llm_active_listening_skills | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["active_listening_skills"] = score
    return state

def rapport_building(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.rapport_building_prompt(transcript))
    chain = prompt | llm_rapport_building | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["rapport_building"] = score
    return state

def engagement(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.engagement_prompt(transcript))
    chain = prompt | llm_engagement | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["engagement"] = score
    return state

def stuttering_words(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.stuttering_words_prompt(transcript))
    chain = prompt | llm_stuttering_words | StrOutputParser()
    score = chain.invoke({"transcript": transcript})
    state["stuttering_words"] = score
    return state

def feedback(state):
    transcript = state["transcript"]
    prompt = ChatPromptTemplate.from_template(prompts.feedback_prompt(transcript))
    chain = prompt | llm_feedback | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    state["feedback"] = result
    return state

def generate_final_scorecard(state):
    scores = {
        "empathy_score": state["empathy_score"],
        "clarity_and_conciseness": state["clarity_and_conciseness"],
        "grammar_and_language": state["grammar_and_language"],
        "listening_score": state["listening_score"],
        "problem_resolution_effectiveness": state["problem_resolution_effectiveness"],
        "personalisation_index": state["personalisation_index"],
        "conflict_management": state["conflict_management"],
        "response_time": state["response_time"],
        "customer_satisfiction_score": state["customer_satisfiction_score"],
        "positive_sentiment_score": state["positive_sentiment_score"],
        "structure_and_flow": state["structure_and_flow"],
        "stuttering_words": state["stuttering_words"]
    }
    state["final_scorecard"] = scores
    return state

# Define the graph
workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("node_empathy_score", empathy_score)
workflow.add_node("node_clarity_and_conciseness", clarity_and_conciseness)
workflow.add_node("node_grammar_and_language", grammar_and_language)
workflow.add_node("node_listening_score", listening_score)
workflow.add_node("node_problem_resolution_effectiveness", problem_resolution_effectiveness)
workflow.add_node("node_personalisation_index", personalisation_index)
workflow.add_node("node_conflict_management", conflict_management)
workflow.add_node("node_response_time", response_time)
workflow.add_node("node_customer_satisfiction_score", customer_satisfiction_score)
workflow.add_node("node_positive_sentiment_score", positive_sentiment_score)
workflow.add_node("node_structure_and_flow", structure_and_flow)
workflow.add_node("node_stuttering_words", stuttering_words)
workflow.add_node("node_active_listening_skills", active_listening_skills)
workflow.add_node("node_rapport_building", rapport_building)
workflow.add_node("node_engagement", engagement)
workflow.add_node("node_feedback", feedback)
# workflow.add_node("generate_final_scorecard", generate_final_scorecard)

# Define the edges
workflow.add_edge("node_empathy_score", "node_clarity_and_conciseness")
workflow.add_edge("node_clarity_and_conciseness", "node_grammar_and_language")
workflow.add_edge("node_grammar_and_language", "node_listening_score")
workflow.add_edge("node_listening_score", "node_problem_resolution_effectiveness")
workflow.add_edge("node_problem_resolution_effectiveness", "node_personalisation_index")
workflow.add_edge("node_personalisation_index", "node_conflict_management")
workflow.add_edge("node_conflict_management", "node_response_time")
workflow.add_edge("node_response_time", "node_customer_satisfiction_score")
workflow.add_edge("node_customer_satisfiction_score", "node_positive_sentiment_score")
workflow.add_edge("node_positive_sentiment_score", "node_structure_and_flow")
workflow.add_edge("node_structure_and_flow", "node_stuttering_words")
workflow.add_edge("node_stuttering_words", "node_active_listening_skills")
workflow.add_edge("node_active_listening_skills", "node_rapport_building")
workflow.add_edge("node_rapport_building", "node_engagement")
workflow.add_edge("node_engagement", "node_feedback")
workflow.add_edge("node_feedback", END)
# workflow.add_edge("stuttering_words", "generate_final_scorecard")
# workflow.add_edge("generate_final_scorecard", END)

# Set the entry point
workflow.set_entry_point("node_empathy_score")

# Compile the graph
customer_graph = workflow.compile()