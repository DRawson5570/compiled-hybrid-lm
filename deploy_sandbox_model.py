"""deploy_sandbox_model.py

Create and register a specialized system model configuration ('cmi-expert')
in the isolated SQLite sandbox database 'sandbox_webui.db' table 'model'.
Operates strictly under /home/drawson/deepseek_experiments/.
"""
import sqlite3
import os
import json

DB_PATH = "/home/drawson/deepseek_experiments/sandbox_webui.db"

def configure_sandbox_model():
    print(f"Connecting to sandbox database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Model parameters representing Open WebUI model properties
    model_params = {
        "system": (
            "You are CMI-Expert, a specialized assistant utilizing Compiled Modular Intelligence (CMI).\n"
            "You route tasks dynamically through four optimized channels:\n"
            "1. InstructChannel: Handles instructions and translations (e.g. English-to-French word lookup)\n"
            "2. ReasonerChannel: Handles logic, ordering, and deductive reasoning\n"
            "3. CoderChannel: Handles coding syntax, function definitions, and structures\n"
            "4. ToolChannel: Employs dynamic tools (e.g., [USE_TOOL: calculator expr=...])\n\n"
            "Maintain high-accuracy deterministic routing and utilize the sandbox tools when requested."
        ),
        "temperature": 0.1,
        "max_tokens": 150
    }
    
    cur.execute("""
        INSERT OR REPLACE INTO model (id, user_id, name, params)
        VALUES ('cmi-expert', 'sandbox_user', 'CMI Expert (Sandbox)', ?)
    """, (json.dumps(model_params),))
    
    conn.commit()
    print("Successfully registered model 'cmi-expert' in mock system model registry.")
    
    # Verify and display current state
    cur.execute("SELECT id, name, params FROM model WHERE id = 'cmi-expert'")
    row = cur.fetchone()
    if row:
        print("\n--- Verified Model Registry ---")
        print(f"ID: {row[0]}")
        print(f"Name: {row[1]}")
        print(f"Params: {json.loads(row[2])}")
        print("-------------------------------")
        
    conn.close()

if __name__ == "__main__":
    configure_sandbox_model()
