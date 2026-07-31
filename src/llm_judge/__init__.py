"""llm_judge — internal-signal prediction of LLM-judge verdict correctness.

Pipeline stages (see scripts/ and docs/DESIGN.md):
  CPU : dataset download -> judge-bank construction -> analysis
  GPU : solver -> judge -> judge+spectral (spectral_trust)
"""

__version__ = "0.3.0"
