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

        # Invoke the appropriate agent based on the type
        if type == "customer":
            result = customer_graph.invoke({"transcript": formatted_transcript})
            result.pop("transcript")
            return result
        elif type == "sales":
            result = sales_graph.invoke({"transcript": formatted_transcript})
            result.pop("transcript")
            return result
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