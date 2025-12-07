###############################################################
# AI Medical Prescription Verification – Single File Version
###############################################################

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st

###############################################################
# --------------------- NLP MODEL LOADING ---------------------
###############################################################

# Load IBM Granite model
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-1b")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.0-h-1b")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

###############################################################
# --------------------- FASTAPI BACKEND ------------------------
###############################################################

app = FastAPI(title="Medical Drug Intelligence API")

class DrugRequest(BaseModel):
    text: str
    age: int


def extract_drug_info(text):
    """Extract drug information using IBM Granite."""
    messages = [
        {"role": "user", "content": f"Extract structured drug details (drug name, dosage, frequency) from this text:\n{text}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])


def check_interactions(drugs):
    """Dummy interaction checker — replace with real drug DB."""
    interactions = []
    if "ibuprofen" in drugs.lower() and "aspirin" in drugs.lower():
        interactions.append("Ibuprofen and Aspirin should not be taken together.")
    return interactions


def recommend_dosage(drug, age):
    """Dummy rule-based dosage (you can connect to a real DB later)."""
    if age < 12:
        return f"Lower pediatric dose recommended for {drug}."
    return f"Standard adult dosage appropriate for {drug}."


def suggest_alternatives(drug):
    """Dummy drug alternative suggestions."""
    alternatives = {
        "ibuprofen": ["acetaminophen", "naproxen"],
        "amoxicillin": ["azithromycin", "cephalexin"],
    }
    return alternatives.get(drug.lower(), ["No alternatives found."])


@app.post("/analyze")
def analyze(request: DrugRequest):
    extracted = extract_drug_info(request.text)

    interactions = check_interactions(extracted)
    dosage_recommendations = recommend_dosage(extracted, request.age)
    alternatives = suggest_alternatives(extracted)

    return {
        "extracted_drug_info": extracted,
        "interactions": interactions,
        "dosage_recommendations": dosage_recommendations,
        "alternatives": alternatives
    }

###############################################################
# -------------------- STREAMLIT FRONTEND ----------------------
###############################################################

def start_streamlit():
    st.title("AI Medical Prescription Verification")

    user_text = st.text_area("Enter prescription text:")
    age = st.number_input("Patient Age", min_value=1, max_value=120)

    if st.button("Analyze"):
        import requests
        payload = {"text": user_text, "age": age}
        response = requests.post("http://localhost:8000/analyze", json=payload)

        if response.status_code == 200:
            data = response.json()
            st.subheader("Extracted Drug Info")
            st.write(data["extracted_drug_info"])

            st.subheader("Detected Interactions")
            st.write(data["interactions"])

            st.subheader("Dosage Recommendations")
            st.write(data["dosage_recommendations"])

            st.subheader("Alternative Medication Suggestions")
            st.write(data["alternatives"])
        else:
            st.error("Error in API call.")

###############################################################
# -------------------- RUN BOTH SERVICES -----------------------
###############################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "ui"], required=True,
                        help="Run FastAPI backend (`api`) or Streamlit UI (`ui`)")
    args = parser.parse_args()

    if args.mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        start_streamlit()
