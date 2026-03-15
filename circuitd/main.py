"""CLI entry point for the circuitd agent."""

import argparse
import logging
import sys
from pathlib import Path

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
        "--backend",
        choices=["ollama", "openai"],
        default=config.BACKEND,
        help=f"LLM backend (default: {config.BACKEND})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: backend-specific default)",
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
    parser.add_argument(
        "--prompts-log",
        metavar="FILE",
        default=None,
        help="Append all prompts and responses to FILE (default: circuitd_prompts.log in cwd)",
    )
    parser.add_argument(
        "--no-prompts-log",
        action="store_true",
        help="Do not write prompts/responses to a file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.parts_url != config.PARTS_API_URL:
        config.PARTS_API_URL = args.parts_url
    if args.no_prompts_log:
        config.PROMPTS_LOG_PATH = None
    elif args.prompts_log is not None:
        config.PROMPTS_LOG_PATH = Path(args.prompts_log)

    run_agent(
        prompt=args.prompt,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        ollama_url=args.ollama_url,
    )
