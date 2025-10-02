"""
Entrypoint: python -m rag_plotting
Purpose:
    Quick manual test runner. Adjust the sample query below or pass one via CLI args.

Usage:
    python -m rag_plotting "make a bar chart of the market cap in stock markets in the Americas"
"""

from __future__ import annotations
import sys
from rag_plotting.pipeline import answer_and_plot

def main():
    query = "please draw a map showing all countries in europe with a T+2 settlement deadline"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    answer_and_plot(query)

if __name__ == "__main__":
    main()
