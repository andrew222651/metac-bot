"""
LightningRod SDK Evaluation Example

Demonstrates the full evaluation pipeline:
  news collection -> question generation -> labeling -> multi-model evaluation -> metrics

Usage:
    uv run --group integrations python integrations/main_lightningrod_eval.py

Requires LIGHTNINGROD_API_KEY in your .env file.
Sign up at https://dashboard.lightningrod.ai to get your API key and $50 of free credits.
More examples available at https://github.com/lightning-rod-labs/lightningrod-python-sdk/tree/main
"""

from datetime import datetime, timedelta
from typing import Annotated

from cyclopts import App, Parameter
from dotenv import load_dotenv
from lightningrod import (
    LightningRod,
    NewsSeedGenerator,
    ForwardLookingQuestionGenerator,
    WebSearchLabeler,
    QuestionPipeline,
    NewsContextGenerator,
    QuestionRenderer,
    RolloutGenerator,
    RolloutScorer,
    BinaryAnswerType,
    Sample,
    ModelConfig,
    open_router_model,
)
from lightningrod.utils import compute_consensus, compute_metrics_summary

import os

app = App(
    name="lightningrod-eval",
    help="Evaluate forecasting models with LightningRod.",
    version_flags=[],
)

# This determines the type of news that will be used to generate questions. It can be a single query or a list of queries.
SEARCH_QUERY = "breaking news"

# Give the question generator specific instructions for the type of questions to generate from the retrieved articles.
# The question generator will also accept a list of good or bad examples or filter criteria.
INSTRUCTIONS = "Generate questions about the provided article. Questions should be clearly resolvable within 1-2 months."

# The type of question you want to generate. This can be binary, multiple choice, continuous, or free response.
ANSWER_TYPE = BinaryAnswerType()

# The pipeline can generate rollouts from any model on openrouter and evaluate predictions against the labeled questions.
# Models are given context about the question from a news search up to the date of the article the question is generated from.
MODELS_TO_EVALUATE = [
    "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
]


def run_news_eval(lr: LightningRod, max_seeds: int) -> list[Sample]:
    """Configure and run the full news evaluation pipeline."""

    # Date range: ~4 months ago to ~2 months ago so questions can be resolved
    end_date = datetime.now() - timedelta(days=60)
    start_date = end_date - timedelta(days=120)

    seed_generator = NewsSeedGenerator(
        start_date=start_date,
        end_date=end_date,
        search_query=SEARCH_QUERY,
    )

    question_generator = ForwardLookingQuestionGenerator(
        instructions=INSTRUCTIONS,
        answer_type=ANSWER_TYPE,
    )

    labeler = WebSearchLabeler(answer_type=ANSWER_TYPE)

    context_generator = NewsContextGenerator()

    renderer = QuestionRenderer(answer_type=ANSWER_TYPE)

    models: list[ModelConfig] = [
        open_router_model(model) for model in MODELS_TO_EVALUATE
    ]

    rollout_generator = RolloutGenerator(models=models)

    scorer = RolloutScorer(answer_type=ANSWER_TYPE)

    pipeline = QuestionPipeline(
        seed_generator=seed_generator,
        question_generator=question_generator,
        context_generators=[context_generator],
        labeler=labeler,
        renderer=renderer,
        rollout_generator=rollout_generator,
        scorer=scorer,
    )

    dataset = lr.transforms.run(
        pipeline, max_seeds=max_seeds, name="News Forecasting Benchmark"
    )

    return dataset.download()


def print_results(samples: list[Sample]) -> None:
    """Print per-model metrics and consensus summary."""
    valid = sum(1 for s in samples if s.rollouts)
    print(
        f"\n{len(samples)} samples, {valid} valid ({valid / len(samples) * 100:.0f}%)\n"
    )

    ranked = sorted(
        compute_metrics_summary(samples).items(),
        key=lambda x: x[1]["mean_reward"],
        reverse=True,
    )
    for rank, (model, m) in enumerate(ranked, 1):
        name = model.split("/")[-1] if "/" in model else model
        print(
            f"  #{rank}  {name:<25} reward={m['mean_reward']:.4f}  parse_rate={m['parse_rate']:.0%}"
        )

    consensus = compute_consensus(samples)
    if consensus:
        n_agree = sum(1 for c in consensus if c["all_agree"])
        print(
            f"\nConsensus: {n_agree}/{len(consensus)} questions agree ({n_agree / len(consensus) * 100:.0f}%)"
        )


@app.default
def main(
    *,
    max_seeds: Annotated[
        int,
        Parameter(
            name=("--max-seeds", "--max-questions"),
            help="Maximum news seeds; --max-questions is a legacy alias.",
        ),
    ] = 20,
) -> None:
    load_dotenv()
    lr = LightningRod(api_key=os.getenv("LIGHTNINGROD_API_KEY"))

    print(f"Running news evaluation with up to {max_seeds} news seeds...\n")
    samples = run_news_eval(lr, max_seeds)
    print_results(samples)


if __name__ == "__main__":
    app()
