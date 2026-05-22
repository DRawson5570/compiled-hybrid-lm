"""openwebui_tool_sandbox.py

Isolated Sandbox copy of the CMI Tool representing capability channels.
Designed to run strictly inside the /home/drawson/deepseek_experiments/ directory.
"""
import sys
import os
import sqlite3
import json

class Tools:
    def __init__(self):
        pass

    def cmi_evaluate_math(self, expression: str) -> str:
        """
        Evaluate an arithmetic math expression deterministically using CMI.
        
        :param expression: mathematical statement, e.g. "12+15"
        :return: evaluation result
        """
        try:
            sanitized = "".join([c for c in expression if c in "0123456789+-*/()"])
            val = int(eval(sanitized))
            return f"Calculated: {val}"
        except Exception as e:
            return f"Error: {str(e)}"

    def cmi_translation_lookup(self, text: str) -> str:
        """
        Lookup a word translation to french using localized sandbox dictionary.
        
        :param text: English word, e.g. "dog"
        :return: French translation
        """
        if text.strip().lower() == "dog":
            return "chien"
        return "Unknown word"
