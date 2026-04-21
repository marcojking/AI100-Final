"""
AI 100 — Homework 4
Use the Claude API (via the Anthropic Python SDK + Vercel AI Gateway) to send prompts
and inspect responses.

Requirements:
    pip install anthropic

Environment variable (set before running):
    AI_GATEWAY_API_KEY=<your Vercel AI Gateway API key>
"""

import os
import anthropic

client = anthropic.Anthropic(
    base_url="https://ai-gateway.vercel.sh",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

MODEL = "anthropic/claude-haiku-4.5"

PROMPTS = [
    {
        "label": "Summarization",
        "user": (
            "Summarize the following in two sentences: "
            "Large language models (LLMs) are neural networks trained on massive text corpora "
            "using a self-supervised objective called next-token prediction. At scale, these "
            "models develop emergent capabilities such as in-context learning, chain-of-thought "
            "reasoning, and instruction following without task-specific fine-tuning."
        ),
    },
    {
        "label": "Reasoning / Math",
        "user": (
            "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? "
            "Show your reasoning step by step."
        ),
    },
    {
        "label": "Creative Writing",
        "user": "Write a two-sentence story about a robot who learns to feel emotions.",
    },
]


def run():
    print("=" * 60)
    print("AI 100 — Homework 4: Interacting with an LLM API via Code")
    print(f"Model: {MODEL}  |  Gateway: Vercel AI Gateway")
    print("=" * 60)

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[Prompt {i}: {prompt['label']}]")
        print(f"Input:  {prompt['user']}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt["user"]}],
        )

        output_text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )

        print(f"Output: {output_text.strip()}")
        print(f"Stop reason: {response.stop_reason}")
        print(f"Tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")
        print("-" * 60)


if __name__ == "__main__":
    run()
