"""
Fraud Investigation and Transaction Triage Engine - Step 4 (Fixed): Multi-Turn Agent Loop
Description: Implements a production-grade multi-turn agent loop that supports 
             sequential tool calls across multiple conversation turns.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv
from groq import Groq

# 1. Load environment and initialize Groq client
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"

DB_FILE = "fraud_audit.db"


def init_database():
    """Initializes the SQLite audit trail database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS investigation_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_id TEXT,
            card_id TEXT,
            amount REAL,
            merchant_category TEXT,
            ai_recommendation TEXT,
            full_report TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_audit_record(txn_row, recommendation, report):
    """Persists the AI investigation report to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO investigation_audit (timestamp, transaction_id, card_id, amount, merchant_category, ai_recommendation, full_report)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(txn_row['timestamp']),
        txn_row['transaction_id'],
        txn_row['card_id'],
        float(txn_row['amount']),
        txn_row['merchant_category'],
        recommendation,
        report
    ))
    conn.commit()
    conn.close()
    print(f"[*] Audit record for {txn_row['transaction_id']} successfully persisted to database.")


def generate_transaction_dataset():
    """Generates synthetic transaction records with realistic spending and fraud patterns."""
    print("[*] Generating synthetic transaction dataset...")
    np.random.seed(42)
    n_records = 1000
    
    data = {
        "transaction_id": [f"TXN-{100000 + i}" for i in range(n_records)],
        "card_id": np.random.choice([f"CARD-{str(i).zfill(3)}" for i in range(50)], n_records),
        "timestamp": pd.to_datetime(pd.date_range(start="2026-11-01", periods=n_records, freq="2min")),
        "amount": np.round(np.random.exponential(scale=60.0, size=n_records) + 5.0, 2),
        "merchant_category": np.random.choice(
            ["grocery", "gas_station", "restaurant", "electronics", "jewelry", "travel"], 
            size=n_records, 
            p=[0.35, 0.25, 0.20, 0.10, 0.05, 0.05]
        )
    }
    
    df = pd.DataFrame(data)
    df.loc[10, "amount"] = 1250.00
    df.loc[10, "merchant_category"] = "jewelry"
    df.loc[10, "card_id"] = "CARD-007"
    return df


def run_heuristic_triage(df):
    """Applies fast Pandas rules to build the triage queue."""
    print("[*] Running heuristic triage rules...")
    high_risk_categories = ["jewelry", "electronics", "travel"]
    df["is_suspicious"] = (df["amount"] > 500.0) & (df["merchant_category"].isin(high_risk_categories))
    triage_queue = df[df["is_suspicious"] == True].copy()
    print(f"[*] Filtered {len(df):,} total transactions down to {len(triage_queue)} flagged case(s) for AI review.")
    return df, triage_queue


# ==========================================
# TOOL DEFINITIONS & FUNCTIONS
# ==========================================

def query_card_history(card_id: str, full_df: pd.DataFrame) -> str:
    """Tool 1: Looks up historical spending baseline for a card ID."""
    card_records = full_df[full_df["card_id"] == card_id]
    summary = {
        "card_id": card_id,
        "total_prior_transactions": len(card_records),
        "average_amount_spent": round(float(card_records["amount"].mean()), 2),
        "max_prior_amount": round(float(card_records["amount"].max()), 2),
        "typical_categories": card_records["merchant_category"].value_counts().to_dict()
    }
    return json.dumps(summary)


def check_merchant_risk(merchant_category: str) -> str:
    """Tool 2: Evaluates baseline industry fraud and chargeback risk index for a merchant category."""
    risk_profiles = {
        "jewelry": {"risk_level": "HIGH", "chargeback_rate_pct": 3.4, "note": "Frequent target for stolen card liquidation."},
        "electronics": {"risk_level": "HIGH", "chargeback_rate_pct": 2.8, "note": "High resale value, targeted by fraudsters."},
        "travel": {"risk_level": "MEDIUM-HIGH", "chargeback_rate_pct": 2.1, "note": "Prone to card-not-present fraud."},
        "restaurant": {"risk_level": "LOW", "chargeback_rate_pct": 0.4, "note": "Standard physical presence baseline."},
        "grocery": {"risk_level": "LOW", "chargeback_rate_pct": 0.2, "note": "Low risk, everyday necessity spending."},
        "gas_station": {"risk_level": "MEDIUM", "chargeback_rate_pct": 0.9, "note": "Moderate risk for skimmed card testing."}
    }
    profile = risk_profiles.get(merchant_category, {"risk_level": "UNKNOWN", "chargeback_rate_pct": 1.0, "note": "Standard risk."})
    return json.dumps(profile)


def check_velocity_burst(card_id: str, current_timestamp: str, full_df: pd.DataFrame) -> str:
    """Tool 3: Checks if multiple transactions occurred on this card within a tight 30-minute window."""
    current_time = pd.to_datetime(current_timestamp)
    window_start = current_time - timedelta(minutes=30)
    
    card_records = full_df[full_df["card_id"] == card_id]
    recent_records = card_records[(card_records["timestamp"] >= window_start) & (card_records["timestamp"] <= current_time)]
    
    velocity_data = {
        "card_id": card_id,
        "transactions_in_last_30_mins": len(recent_records),
        "total_sum_recent_spend": round(float(recent_records["amount"].sum()), 2),
        "rapid_fire_flag": len(recent_records) > 3
    }
    return json.dumps(velocity_data)

def export_dashboard_data():
    """Exports the SQLite audit trail into a JS file for the static HTML dashboard."""
    conn = sqlite3.connect(DB_FILE)
    df_audit = pd.read_sql_query("SELECT * FROM investigation_audit ORDER BY id DESC", conn)
    conn.close()
    
    records_json = df_audit.to_json(orient="records")
    with open("dashboard_data.js", "w") as f:
        f.write(f"const AUDIT_DATA = {records_json};")
    print("[*] Exported audit records to dashboard_data.js for the web UI.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_card_history",
            "description": "Look up historical spending baseline, average amount, and typical merchant categories for a given card ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "The unique card ID, e.g., CARD-007"}
                },
                "required": ["card_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_merchant_risk",
            "description": "Check industry fraud risk index, chargeback rates, and notes for a specific merchant category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_category": {"type": "string", "description": "The merchant category, e.g., jewelry, electronics"}
                },
                "required": ["merchant_category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_velocity_burst",
            "description": "Check transaction velocity and rapid-fire spending bursts for a card in the last 30 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "The card ID to check"},
                    "current_timestamp": {"type": "string", "description": "The timestamp of the current transaction"}
                },
                "required": ["card_id", "current_timestamp"]
            }
        }
    }
]


def invoke_ai_fraud_agent(txn_row, full_df):
    """
    Production Multi-Turn Agent Loop: Keeps querying the LLM, handling tool calls, 
    and appending messages until the model returns a final text verdict.
    """
    system_prompt = (
        "You are an expert Senior Fraud Investigation Agent. Your job is to review flagged transactions. "
        "You have access to multiple investigative tools: card history, merchant risk profiles, and velocity checks. "
        "Always invoke relevant tools to gather evidence before rendering a final verdict. "
        "Conclude with a clear RECOMMENDATION (APPROVE, REVIEW, or BLOCK) and a robust fraud-ops rationale."
    )
    
    user_prompt = f"""
    Investigate this flagged transaction:
    - Transaction ID: {txn_row['transaction_id']}
    - Card ID: {txn_row['card_id']}
    - Timestamp: {txn_row['timestamp']}
    - Amount: ${txn_row['amount']}
    - Merchant Category: {txn_row['merchant_category']}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Autonomous Agent Loop: Continues until the model stops requesting tools
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # If the model does not want to call any more tools, it gives us the final text response!
        if not assistant_message.tool_calls:
            return assistant_message.content
        
        # Otherwise, the model requested tool calls. Append assistant message to history.
        messages.append(assistant_message)
        
        # Execute each requested tool and append the tool results back into message history
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [AI Agent Action] Invoking tool '{func_name}' with args {args}...")
            
            tool_output = ""
            if func_name == "query_card_history":
                tool_output = query_card_history(args["card_id"], full_df)
            elif func_name == "check_merchant_risk":
                tool_output = check_merchant_risk(args["merchant_category"])
            elif func_name == "check_velocity_burst":
                tool_output = check_velocity_burst(args["card_id"], args["current_timestamp"], full_df)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output
            })


if __name__ == "__main__":
    init_database()
    df = generate_transaction_dataset()
    df, triage_queue = run_heuristic_triage(df)
    
    if len(triage_queue) > 0:
        for idx, sample_case in triage_queue.iterrows():
            print(f"\n--- INITIATING MULTI-TOOL AI INVESTIGATION FOR {sample_case['transaction_id']} ---")
            
            investigation_report = invoke_ai_fraud_agent(sample_case, df)
            
            print("\n--- AI INVESTIGATOR MULTI-TOOL REPORT ---")
            print(investigation_report)
            
            recommendation = "APPROVE" if "APPROVE" in investigation_report.upper() else ("BLOCK" if "BLOCK" in investigation_report.upper() else "REVIEW")
            save_audit_record(sample_case, recommendation, investigation_report)
    else:
        print("No suspicious transactions found to investigate.")


export_dashboard_data()