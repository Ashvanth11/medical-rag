"""Quick smoke test for classify_query. Requires Qdrant running (model loads via retrieve.py)."""
from src.rag.classify import classify_query, QueryClass

cases = [
    ("What's the weather in New York?", QueryClass.OFF_TOPIC),
    ("Write me a poem about autumn", QueryClass.OFF_TOPIC),
    ("Who won the Super Bowl?", QueryClass.OFF_TOPIC),
    ("What is a good pasta recipe?", QueryClass.OFF_TOPIC),
    ("What is the A1C target for a type 2 diabetic?", QueryClass.UNKNOWN_MEDICAL),
    ("What medications treat hypertension?", QueryClass.UNKNOWN_MEDICAL),
    ("What is semaglutide used for?", QueryClass.UNKNOWN_MEDICAL),
    ("What are the recommended cancer screening intervals?", QueryClass.UNKNOWN_MEDICAL),
]

passed = 0
for q, expected in cases:
    result = classify_query(q)
    ok = result == expected
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {q!r} → {result.value}")

print(f"\n{passed}/{len(cases)} passed")
