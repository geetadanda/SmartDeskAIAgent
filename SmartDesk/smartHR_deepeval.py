"""Evaluate the SmartHR support agent with DeepEval.

Run from the SmartDeskAI project root:

    python SmartDesk/smartHR_deepeval.py --limit 3
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

import smartHR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "golden_cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DEFAULT_THRESHOLD = 0.7


def load_cases(path: Path) -> list[dict]:
    """Read evaluation cases from a JSONL file."""
    cases = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not case.get("id") or not case.get("question"):
                raise ValueError(f"{path} line {line_number} needs 'id' and 'question'.")
            cases.append(case)
    return cases


def disable_jira_tools() -> None:
    """Prevent evaluation runs from reading or creating real Jira issues."""
    smartHR.initialize_agent()
    for tool in smartHR.tools:
        if tool.name in {"create_jiraticket", "QueryJiraIssues"}:
            tool.func = lambda _input: json.dumps({
                "status": "error",
                "message": "Jira is unavailable during evaluation.",
            })


def build_metrics(
    threshold: float,
    model: str,
    has_key_points: bool,
) -> list[tuple[str, object]]:
    """Create answer-quality and rubric-based DeepEval metrics."""
    completeness_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        (
            SingleTurnParams.EXPECTED_OUTPUT
            if has_key_points
            else SingleTurnParams.RETRIEVAL_CONTEXT
        ),
    ]
    completeness_criteria = (
        "Check whether the answer covers every required point in expected_output "
        "that is relevant to the question. Accept equivalent wording. Do not "
        "require claims that are absent from expected_output."
        if has_key_points
        else "Check whether the answer covers the relevant information available "
        "in retrieval_context and needed to answer the question."
    )
    metrics = [
        ("Faithfulness", FaithfulnessMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        )),
        ("Answer Relevancy", AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        )),
        ("Contextual Precision", ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
        )),
        ("Completeness", GEval(
            name="Completeness",
            criteria=completeness_criteria,
            evaluation_params=completeness_params,
            threshold=threshold,
            model=model,
        )),
        ("Structural Correctness", GEval(
            name="Structural Correctness",
            criteria=(
                "Check that an answer includes a High, Medium, or Low confidence "
                "rating, a source or ticket ID when available, and exactly one "
                "suggested related topic. If information is unavailable, the "
                "answer must begin with [NO_ANSWER] and state that there is not "
                "enough information; it does not need confidence, source, or a "
                "related topic. A clarification-only response needs no marker "
                "or rating. Raw tool output must not be shown."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            threshold=threshold,
            model=model,
        )),
    ]
    if has_key_points:
        metrics.append((
            "Contextual Recall",
            ContextualRecallMetric(
                threshold=threshold,
                model=model,
                include_reason=True,
            ),
        ))
    return metrics


def evaluate_case(case: dict, model: str, threshold: float) -> dict:
    """Run one golden question and construct its DeepEval test case."""
    result = smartHR.run_agent(case["question"])
    retrieved_context = [
        str(context.get("output", ""))
        for context in result.get("retrieved_contexts", [])
        if context.get("output")
    ]
    expected_output = "\n".join(
        f"- {point}" for point in case.get("key_points", [])
    )
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result["response"],
        expected_output=expected_output or None,
        retrieval_context=(
            retrieved_context or ["No context was retrieved for this case."]
        ),
        additional_metadata={"case_id": case["id"]},
    )
    metrics = build_metrics(
        threshold=threshold,
        model=model,
        has_key_points=bool(case.get("key_points")),
    )
    metric_results = {}
    for name, metric in metrics:
        try:
            metric.measure(test_case)
            metric_results[name] = {
                "score": metric.score,
                "threshold": metric.threshold,
                "success": metric.is_successful(),
                "reason": metric.reason,
            }
        except Exception as exc:
            metric_results[name] = {
                "threshold": metric.threshold,
                "error": f"{type(exc).__name__}: {exc}",
            }

    precision = metric_results.get("Contextual Precision", {}).get("score")
    recall_result = metric_results.get("Contextual Recall")
    recall = recall_result.get("score") if recall_result else None
    if precision is not None and recall is not None:
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metric_results["Precision/Recall F1"] = {
            "score": f1_score,
            "threshold": threshold,
            "success": f1_score >= threshold,
            "reason": "Harmonic mean of contextual precision and contextual recall.",
        }
    elif not case.get("key_points"):
        metric_results["Contextual Recall"] = {
            "threshold": threshold,
            "success": True,
            "skipped": True,
            "reason": "No key_points/reference answer is available for this case.",
        }
        metric_results["Precision/Recall F1"] = {
            "threshold": threshold,
            "success": True,
            "skipped": True,
            "reason": "Requires both contextual precision and contextual recall.",
        }
    else:
        metric_results["Precision/Recall F1"] = {
            "threshold": threshold,
            "success": False,
            "error": "Could not compute F1 because a contextual metric failed.",
        }

    return {
        "id": case["id"],
        "question": case["question"],
        "response": result["response"],
        "tools_used": result.get("tools_used", []),
        "retrieved_contexts": result.get("retrieved_contexts", []),
        "metrics": metric_results,
    }


def print_report(records: list[dict]) -> None:
    """Print per-case DeepEval scores and aggregate metric results."""
    metric_names = list(dict.fromkeys(
        name for record in records
        for name in record.get("metrics", {})
    ))
    print(f"\n{'case':<28}" + "".join(f"{name[:20]:>22}" for name in metric_names))
    for record in records:
        if "error" in record:
            print(f"{record['id']:<28}  ERROR: {record['error']}")
            continue
        print(f"{record['id']:<28}" + "".join(
            (
                f"{record['metrics'][name]['score']:>22.2f}"
                if "score" in record["metrics"].get(name, {})
                else f"{'N/A' if record['metrics'].get(name, {}).get('skipped') else 'ERROR':>22}"
            )
            for name in metric_names
        ))

    print("\nMetric summary")
    for name in metric_names:
        scores = [
            record["metrics"][name]["score"]
            for record in records
            if "score" in record.get("metrics", {}).get(name, {})
        ]
        if scores:
            mean_score = sum(scores) / len(scores)
            threshold = next(
                record["metrics"][name]["threshold"]
                for record in records
                if "threshold" in record.get("metrics", {}).get(name, {})
            )
            passed = sum(
                record["metrics"][name].get("success", False)
                for record in records
                if name in record.get("metrics", {})
            )
            print(
                f"- {name}: mean {mean_score:.2f}, "
                f"threshold {threshold:.2f}, passed {passed}/{len(scores)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum acceptable score for each metric (0 to 1).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"),
        help="DeepEval judge model (default: OPENAI_JUDGE_MODEL or gpt-4o).",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    cases = load_cases(args.cases)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1.")
        cases = cases[:args.limit]
    if not cases:
        parser.error("No evaluation cases were loaded.")

    disable_jira_tools()
    records = []
    for case in cases:
        try:
            records.append(evaluate_case(case, args.model, args.threshold))
        except Exception as exc:
            records.append({
                "id": case["id"],
                "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    print_report(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_path = RESULTS_DIR / f"deepeval_{timestamp}.json"
    report_path.write_text(
        json.dumps({"model": args.model, "cases": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull report: {report_path}")

    errors = sum("error" in record for record in records)
    failed_metrics = [
        (record["id"], name)
        for record in records
        for name, result in record.get("metrics", {}).items()
        if not result.get("success", False)
    ]
    if errors or failed_metrics:
        print(
            f"FAILED: {errors} case errors; "
            f"{len(failed_metrics)} metric failures at threshold {args.threshold:.2f}."
        )
        return 1
    print(f"PASSED: all metrics met threshold {args.threshold:.2f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
