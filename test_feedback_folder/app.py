from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
# from main import graph_app
from customer_agent import customer_graph
from sales_agent import sales_graph
import uvicorn
import firebase_admin
from firebase_admin import credentials,firestore
import os
from dotenv import load_dotenv
load_dotenv()


cred = credentials.Certificate(os.getenv("CRED_PATH"))
firebase_admin.initialize_app(cred)

db = firestore.client()


app = FastAPI()
# Define Pydantic model for request body
class TranscriptionRequest(BaseModel):
    transcription: str


import json

def generate_scorecard(transcript, type):
    try:
        # Parse the transcript string into a Python object
        if isinstance(transcript, str):
            transcript = json.loads(transcript)

        # Separate system role (context) from the rest
        context = ""
        conversation = []
        for entry in transcript:
            if entry["role"] == "system":
                context = entry["content"]
            else:
                conversation.append(f"{entry['role']}: {entry['content']}")

        # Combine context and conversation
        formatted_transcript = f"Context:\n{context}\n\nConversation:\n" + "\n".join(conversation)
        print(formatted_transcript)

        # Invoke the appropriate agent based on the type
        if type == "customer":
            result = customer_graph.invoke({"transcript": formatted_transcript})
            result.pop("transcript")
            return {
    "communication_and_delivery" : {
        "empathy_score" : result["empathy_score"],
        "clarity_and_conciseness" : result["clarity_and_conciseness"],
        "grammar_and_language" : result["grammar_and_language"],
        "listening_score" : result["listening_score"],
        "positive_sentiment_score" : result["positive_sentiment_score"],
        "structure_and_flow" : result["structure_and_flow"],
        "stuttering_words" : result["stuttering_words"],   
        "active_listening_skills" : result["active_listening_skills"]
    },
    "customer_interaction_and_resolution" : {
        "problem_resolution_effectiveness" : result["problem_resolution_effectiveness"],
        "personalisation_index" : result["personalisation_index"],
        "conflict_management" : result["conflict_management"],
        "response_time" : result["response_time"],
        "customer_satisfiction_score" : result["customer_satisfiction_score"],
        "rapport_building" : result["rapport_building"],
        "engagement" : result["engagement"]
    },
    "sales_and_persuasion" : {
        "product_knowledge_score": None,
        "persuasion_and_negotiation_skills" : None,
        "objection_handling" : None,
        "upselling_sucess_rate" : None,
        "call_to_action_effectiveness" : None,
        "questioning_technique" : None
    },
    "professionalism_and_presentation" : {
        "confidence_score" : None,
        "value_proposition" : None,
        "pitch_quality" : None
    },
    "feedback" : result["feedback"]
}
        elif type == "sales":
            result = sales_graph.invoke({"transcript": formatted_transcript})
            result.pop("transcript")
            return {
    "communication_and_delivery" : {
        "empathy_score" : None,
        "clarity_and_conciseness" : None,
        "grammar_and_language" : None,
        "listening_score" : None,
        "positive_sentiment_score" : None,
        "structure_and_flow" : None,
        "stuttering_words" : None,
        "active_listening_skills" : None
    },
    "customer_interaction_and_resolution" : {
        "problem_resolution_effectiveness" : None,
        "personalisation_index" : None,
        "conflict_management" : None,
        "response_time" : None,
        "customer_satisfiction_score" : None,
        "rapport_building" : None,
        "engagement" : None
    },
    "sales_and_persuasion" : {
        "product_knowledge_score": result["product_knowledge_score"],
        "persuasion_and_negotiation_skills" : result["persuasion_and_negotiation_skills"],
        "objection_handling" : result["objection_handling"],
        "upselling_sucess_rate" : result["upselling_sucess_rate"],
        "call_to_action_effectiveness" : result["call_to_action_effectiveness"],
        "questioning_technique" : result["questioning_technique"]
    },
    "professionalism_and_presentation" : {
        "confidence_score" : result["confidence_score"],
        "value_proposition" : result["value_proposition"],
        "pitch_quality" : result["pitch_quality"]
    },
    "feedback" : result["feedback"]
}
        else:
            raise ValueError("Invalid type provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define API endpoint
@app.post("/get_scorecard")
async def get_transcription(room_id: str):
    try:
        doc_ref = db.collection(u'Transcription').document(room_id)
        doc = doc_ref.get()
        if doc.exists:
            transcript = doc.to_dict()["transcript"]
            type = doc.to_dict()["type"]

            # Ensure type is valid
            if type in ["customer", "sales"]:
                return generate_scorecard(transcript, type)
            else:
                return {"error": "Invalid type"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transcription not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve transcription: {str(e)}"
        )



# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app,  port=8000)