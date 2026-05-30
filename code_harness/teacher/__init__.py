"""Teacher-driven code improvement pipeline.

Phases:
  1. analyze   — DeepSeek audits target model failures, categorizes weaknesses
  2. document  — Generate per-weakness markdown documentation
  3. synthesize — DeepSeek generates targeted training data per weakness
  4. train     — Two-phase cartridge training (canonical CE → RFT)
  5. integrate — Mount all cartridges, full regression test
  6. pipeline  — End-to-end orchestrator
"""
