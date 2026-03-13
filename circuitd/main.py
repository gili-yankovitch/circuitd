"""CLI entry point for the circuitd agent."""

import argparse
import logging
import sys

from . import config
from .agent import run_agent


def main():
    parser = argparse.ArgumentParser(
        prog="circuitd",
        description="AI agent that designs electronic circuits and outputs .decl files",
    )
    parser.add_argument(
        "prompt",
        help='Circuit description, e.g. "3.3V MCU board with SPI flash and status LED"',
    )
    parser.add_argument(
        "-o", "--output",
        default="output.decl",
        help="Output .decl file path (default: output.decl)",
    )
    parser.add_argument(
        "--model",
        default=config.OLLAMA_MODEL,
        help=f"Ollama model name (default: {config.OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=config.OLLAMA_URL,
        help=f"Ollama server URL (default: {config.OLLAMA_URL})",
    )
    parser.add_argument(
        "--parts-url",
        default=config.PARTS_API_URL,
        help=f"Parts search API URL (default: {config.PARTS_API_URL})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.parts_url != config.PARTS_API_URL:
        config.PARTS_API_URL = args.parts_url

    run_agent(
        prompt=args.prompt,
        output_path=args.output,
        model=args.model,
        ollama_url=args.ollama_url,
    )
