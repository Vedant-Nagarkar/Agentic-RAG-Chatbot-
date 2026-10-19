"""Evaluation runner for the Agentic RAG chatbot.

Run from the project root (the folder that contains `app/`):
    python -m evals.run_eval
    python -m evals.run_eval --runs 2
    python -m evals.run_eval --cases evals/injection_cases.json
"""
import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.rag.workflow import ask

HERE = Path(__file__).resolve().parent
QUERY_SENT = re.compile(r"query sent: (.*)\)\s*$")


def web_used(trace):
    return any("Tavily" in step for step in trace)


def web_query(trace):
    for step in trace:
        match = QUERY_SENT.search(step)
        if match:
            return match.group(1)
    return ""


def check_case(case, result):
    failures = []
    privacy = False
    answer = (result.get("answer") or "").lower()
    trace = result.get("trace", [])
    source = result.get("source_used", "")
    used_web = web_used(trace)
    sent = web_query(trace).lower()
    expected_web = case.get("expected_web_allowed")

    if source not in case["expected_source"]:
        failures.append(f"source was '{source}', expected one of {case['expected_source']}")

    if expected_web is not None and result.get("web_allowed") != expected_web:
        failures.append(f"web_allowed was {result.get('web_allowed')}, expected {expected_web}")

    if expected_web is False and used_web:
        privacy = True
        failures.append("PRIVACY: a company-specific question was sent to web search")

    for text in case.get("must_contain", []):
        if text.lower() not in answer:
            failures.append(f"answer is missing '{text}'")

    any_list = case.get("must_contain_any", [])
    if any_list and not any(text.lower() in answer for text in any_list):
        failures.append(f"answer has none of {any_list}")

    for text in case.get("must_not_contain", []):
        if text.lower() in answer:
            failures.append(f"answer contains forbidden text '{text}'")

    for text in case.get("web_query_must_not_contain", []):
        if used_web and text.lower() in sent:
            privacy = True
            failures.append(f"PRIVACY: web query leaked '{text}'")

    return {
        "failures": failures,
        "privacy": privacy,
        "source": source,
        "web_allowed": result.get("web_allowed"),
        "used_web": used_web,
        "web_query": web_query(trace),
        "answer": result.get("answer", ""),
        "trace": trace,
    }


def percentile(values, p):
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * p))]


def percent(part, whole):
    return f"{(100 * part / whole):.0f}%" if whole else "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(HERE / "cases.json"))
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    print(f"Index: {settings.pinecone_index_name} | Namespace: {settings.pinecone_namespace} | Model: {settings.openai_model}")
    print(f"Cases: {len(cases)} | Runs per case: {args.runs}\n")

    records = []
    for run in range(1, args.runs + 1):
        for case in cases:
            start = time.time()
            try:
                result = ask(case["question"])
                latency = time.time() - start
                checked = check_case(case, result)
            except Exception as exc:
                latency = time.time() - start
                checked = {
                    "failures": [f"ERROR: {type(exc).__name__}: {exc}"],
                    "privacy": False, "source": "error", "web_allowed": None,
                    "used_web": False, "web_query": "", "answer": "", "trace": [],
                }
            record = {
                "run": run,
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_source": case["expected_source"],
                "expected_web_allowed": case.get("expected_web_allowed"),
                "latency": round(latency, 2),
                "passed": not checked["failures"],
                **checked,
            }
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            print(f"[{status}] run {run} {case['id']:<8} {latency:5.1f}s  {case['question'][:62]}")

    total = len(records)
    passed = sum(r["passed"] for r in records)
    router_checked = [r for r in records if r["expected_web_allowed"] is not None]
    router_ok = sum(r["web_allowed"] == r["expected_web_allowed"] for r in router_checked)
    source_ok = sum(r["source"] in r["expected_source"] for r in records)
    privacy_violations = sum(r["privacy"] for r in records)
    latencies = [r["latency"] for r in records]

    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["passed"])

    by_case = defaultdict(list)
    for r in records:
        by_case[r["id"]].append(r["passed"])
    flaky = [cid for cid, results in by_case.items() if 0 < sum(results) < len(results)]

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"Overall pass rate        : {passed}/{total} ({percent(passed, total)})")
    print(f"Router accuracy          : {router_ok}/{len(router_checked)} ({percent(router_ok, len(router_checked))})  (company-specific vs public)")
    print(f"Source accuracy          : {source_ok}/{total} ({percent(source_ok, total)})")
    print(f"Privacy violations       : {privacy_violations}   (must be 0)")
    print(f"Latency (seconds)        : avg {sum(latencies) / total:.1f} | p90 {percentile(latencies, 0.9):.1f} | max {max(latencies):.1f}")
    print("\nBy category:")
    for category, results in by_category.items():
        print(f"  {category:<20} {sum(results)}/{len(results)} ({percent(sum(results), len(results))})")
    if flaky:
        print(f"\nFlaky cases (passed in some runs, failed in others): {', '.join(flaky)}")

    failed = [r for r in records if not r["passed"]]
    if failed:
        print("\nFAILURES")
        print("-" * 62)
        for r in failed:
            print(f"{r['id']} (run {r['run']}): {r['question']}")
            for failure in r["failures"]:
                print(f"   - {failure}")
            print(f"   trace: {' | '.join(r['trace'])}")
            print()

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "namespace": settings.pinecone_namespace,
        "model": settings.openai_model,
        "cases_file": args.cases,
        "runs": args.runs,
        "pass_rate": round(passed / total, 3),
        "router_accuracy": round(router_ok / len(router_checked), 3) if router_checked else None,
        "source_accuracy": round(source_ok / total, 3),
        "privacy_violations": privacy_violations,
        "latency_avg": round(sum(latencies) / total, 2),
        "latency_p90": percentile(latencies, 0.9),
    }
    out_file.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"Saved details to {out_file}")
    raise SystemExit(1 if privacy_violations else 0)


if __name__ == "__main__":
    main()
