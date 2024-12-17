from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uuid
import firebase_admin
from firebase_admin import firestore, credentials
import os
import uvicorn
import random
from dotenv import load_dotenv
load_dotenv()

# Initialize Firebase
cred = credentials.Certificate(os.getenv("CRED_PATH"))
firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scenarios", status_code=status.HTTP_201_CREATED)
async def create_scenario(name: str, prompt: str, type: str, AI_persona: str):
    id = str(uuid.uuid4())
    try:
        doc_ref = db.collection(u'scenarios').document(id)
        doc_ref.set({
            u'name': name,
            u'prompt': prompt,
            u'type': type,
            u'persona':AI_persona
        })
        return {"message": f"Scenario created successfully", "id": id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scenario: {str(e)}"
        )


@app.get("/scenarios/{scenario_id}")
async def get_scenario(roleplay_type: str, difficulty_level: str):
        
    # Query Firestore to get all scenarios for specific rolepay type
    scenarios_ref = db.collection(u'scenarios')
    query = scenarios_ref.where("type", "==", roleplay_type)
    docs = query.stream()
    
    # Collect scenario IDs where the type is "sales"
    sales_scenarios = []
    for doc in docs:
        sales_scenarios.append(doc.id)  # Adding the document ID to the list

    selected_scenario = random.choice(sales_scenarios)

    doc_ref = db.collection(u'scenarios').document(selected_scenario)
    doc = doc_ref.get()
    if doc.exists:
        scenario = doc.to_dict()
        if difficulty_level == "easy":
            return {"name": scenario.get('name', ''),"prompt": scenario.get("easy_prompt",""),"persona_name": scenario.get('persona_name', ''),"persona": scenario.get('persona', ''),"difficulty_level":difficulty_level,"image_url": scenario.get('image_url', ''),"voice_id": scenario.get('voice_id', ''),"type": scenario.get('type','')}
        elif difficulty_level == "medium":
            return {"name": scenario.get('name', ''),"prompt": scenario.get("medium_prompt",""),"persona_name": scenario.get('persona_name', ''),"persona": scenario.get('persona', ''),"difficulty_level":difficulty_level,"image_url": scenario.get('image_url', ''),"voice_id": scenario.get('voice_id', ''),"type": scenario.get('type','')}
        elif difficulty_level == "hard":
            return {"name": scenario.get('name', ''),"prompt": scenario.get("hard_prompt",""),"persona_name": scenario.get('persona_name', ''),"persona": scenario.get('persona', ''),"difficulty_level":difficulty_level,"image_url": scenario.get('image_url', ''),"voice_id": scenario.get('voice_id', ''),"type": scenario.get('type','')}

    
@app.put("/scenarios/{scenario_id}")
async def update_scenario(scenario_id: str, name: str, prompt: str, type: str, AI_persona:str):
    try:
        doc_ref = db.collection(u'scenarios').document(scenario_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scenario not found"
            )
        doc_ref.update({
            u'name': name,
            u'prompt': prompt,
            u'type': type,
            u'persona': AI_persona
        })
        return {"message": "Scenario updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update scenario: {str(e)}"
        )

@app.delete("/scenarios/{scenario_id}")
async def delete_scenario(scenario_id: str):
    try:
        doc_ref = db.collection(u'scenarios').document(scenario_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scenario not found"
            )
        doc_ref.delete()
        return {"message": "Scenario deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete scenario: {str(e)}"
        )
    

@app.get("/scenarios")
async def get_all_scenario_ids():
    try:
        docs = db.collection(u'scenarios').stream()
        
        scenario_ids = [doc.id for doc in docs]
        
        if scenario_ids:
            return {"scenario_ids": scenario_ids}
        else:
            return {"message": "No scenarios found"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scenarios: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0" , port=8080, reload=True)