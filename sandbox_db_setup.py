"""sandbox_db_setup.py

Helper to build and maintain an isolated mock SQLite database 'sandbox_webui.db'
strictly inside /home/drawson/deepseek_experiments/ to validate sandbox tool updates.
"""
import sqlite3
import os
import json
import inspect
from openwebui_tool_sandbox import Tools

DB_PATH = "/home/drawson/deepseek_experiments/sandbox_webui.db"

def extract_specs(cls):
    """Generates Open-WebUI style function declarations automatically."""
    specs = []
    methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]
    
    for m in methods:
        func = getattr(cls, m)
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        args = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            args.append({
                "name": name,
                "type": "string" if param.annotation == str else "unknown",
                "required": param.default == inspect.Parameter.empty
            })
            
        specs.append({
            "name": m,
            "description": doc.split("\n")[0] if doc else "",
            "parameters": args
        })
    return json.dumps(specs)

def setup_sandbox():
    print(f"Creating isolated sandbox database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Create the exact columns representing Open-WebUI schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tool (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            specs TEXT NOT NULL,
            meta TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            valves TEXT,
            access_control JSON
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            params TEXT NOT NULL
        )
    """)
    
    conn.commit()
    
    # Insert cmi_tool inside the local database
    tool_content = open("/home/drawson/deepseek_experiments/openwebui_tool_sandbox.py").read()
    tool_specs = extract_specs(Tools)
    
    cur.execute("""
        INSERT OR REPLACE INTO tool (id, user_id, name, content, specs, meta, created_at, updated_at)
        VALUES ('cmi_tool_sandbox', 'sandbox', 'CMI Sandboxed Tool', ?, ?, '{}', 1700000000, 1700000000)
    """, (tool_content, tool_specs))
    
    print("Registered tool 'cmi_tool_sandbox' in mock sqlite DB.")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_sandbox()
