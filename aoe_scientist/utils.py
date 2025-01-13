"""Utility functions for AI Research Idea Generator and Reviewer."""

import argparse
import os

def setup_argparse():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description='AI Research Idea Generator and Reviewer')
    parser.add_argument('--mode', choices=['generate', 'review'], default='generate',
                       help='Operation mode: generate new ideas or review existing ones')
    parser.add_argument('--idea-path', help='Path to idea JSON file for review mode')
    parser.add_argument('--gen-model', default="gpt-4o",
                       help='Model for generation (default: gpt-4o)')
    parser.add_argument('--review-model', default="deepseek-chat",
                       help='Model for review (default: deepseek-chat)')
    return parser.parse_args()

