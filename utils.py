# utils.py

import faiss
from sentence_transformers import SentenceTransformer
from ollama import Client
import logging
import datetime
import json
import torch.nn as nn
import pandas as pd

# --- RAG Setup ---
fraud_knowledge_base = [
    "High-value transactions from a new device often indicate account takeover.",
    "A fraud ring was detected where multiple cards were used on the same IP address in a short time.",
    "Policy: Transactions over $1500 are flagged for manual review.",
]
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_embeddings = embedding_model.encode(fraud_knowledge_base)
index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
index.add(knowledge_embeddings)
ollama_client = Client(host='http://localhost:11434')

def get_rag_explanation(transaction_id, xgb_score, nn_score, transaction_data):
    query_text = f"Transaction flagged with XGBoost score {xgb_score:.2f} and amount {transaction_data['Amount']:.2f}."
    query_embedding = embedding_model.encode([query_text])
    distances, indices = index.search(query_embedding, k=2)
    retrieved_context = [fraud_knowledge_base[i] for i in indices[0]]
    prompt = f"""
    You are a fraud detection analyst. Analyze the following transaction and provide a clear, concise explanation for why it was flagged.
    Transaction Details: {transaction_data.to_dict()}
    Primary Model Risk Score: {xgb_score:.4f}
    Deep Learning Model Risk Score: {nn_score:.4f}
    Relevant Context: {retrieved_context}
    Provide a final recommended action.
    """
    try:
        response = ollama_client.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error with local LLM: {e}"

# --- Audit Log Setup ---
logging.basicConfig(filename='audit_log.json', level=logging.INFO, format='%(message)s')
def create_audit_log_entry(transaction_id, status, explanation, models_used, risk_score):
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "transaction_id": transaction_id,
        "status": status,
        "risk_score": risk_score,
        "models_used": models_used,
        "explanation": explanation
    }
    logging.info(json.dumps(log_entry))