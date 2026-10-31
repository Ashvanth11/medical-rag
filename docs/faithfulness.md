# Faithfulness evaluation

The judge measures whether generated factual claims are supported by the exact
context supplied to Llama. This is not medical accuracy, completeness, citation
correctness, or clinical safety. Keyword matching remains a separate heuristic.

## Capture answers

Start the existing Qdrant and Ollama services. From the project root, capture
a six-answer pilot spanning topics (each command below creates local artifacts):

```bash
.venv/bin/python -m scripts.evaluate --ids hbp_001 diabetes_001 cholesterol_001 obesity_001 stroke_001 copd_001 --output data/eval/pilot_answers.json
```

This saves full answers, exact post-deduplication context, prompts, retrieval
scores, generation settings, and explicit answer/abstention status. Existing
preview-only results cannot be judged. Capture is checkpointed after each answer;
rerunning capture regenerates answers. Use distinct output filenames to preserve runs.

## Judge the saved answers

To use the funded Anthropic account instead, save `ANTHROPIC_API_KEY=your-key`
in `.env`, then run:

```bash
.venv/bin/python -m scripts.evaluate_faithfulness --provider anthropic --input data/eval/pilot_answers.json --output data/eval/pilot_faithfulness_anthropic.json
```

Anthropic defaults to `claude-sonnet-5-5`, with `between_tools` thinking and medium
effort. It uses structured JSON output and the existing `requests` dependency.
Questions, answers, and excerpts are sent to Anthropic and billed to that account.
Use a separate output file so Gemini and Claude verdicts are never mixed. Token
usage, returned model IDs, per-call timings, and error codes are saved. This is a
fresh judge run over the existing capture; Llama answers are not regenerated.

Run in the terminal where you exported `GEMINI_API_KEY`. Do not paste the key into
source files or commit it. The script reads the environment and never logs the key.
Alternatively save `GEMINI_API_KEY=your-key` in the project's ignored `.env` file;
the evaluator reads that key without executing the file as shell code.

```bash
.venv/bin/python -m scripts.evaluate_faithfulness --input data/eval/pilot_answers.json --output data/eval/pilot_faithfulness.json
```

The default judge is `gemini-3.8-flash`; override with `--model`. Your API project
determines billing and quota: this script cannot enforce that a key is on the free
tier. It sends questions, answers, and chunk text to Google. There are two calls
per answered question, plus one retry per call by default. Requests are sequential
with a default 15-second minimum spacing for Gemini or one second for Anthropic;
increase `--request-delay` for your quota. `--request-timeout` defaults to 45 seconds
and `--attempts` to two. Gemini SDK retries are disabled so they do not multiply
the evaluator's retries. Anthropic uses 10-second connect and 45-second read timeouts.
Permanent API errors are not retried. Judge failures are recorded without scores.

Read the generated `.md` alongside the JSON. Manually review all six pilot answers:
check for omitted or merged claims, lost quantities/qualifiers, and whether the
quoted passages actually entail each supported claim. Exact quote validation
checks provenance only; it cannot prove semantic support. Double-escaped Unicode
in a judge quote is decoded only when the decoded quote matches the chunk text
exactly; the original quote is retained as `raw_quote`. Separately test a copy
of pilot data with an altered number and an added unsupported assertion. Label
these as synthetic controls and exclude them from the benchmark. Freeze prompts
after calibration; don't tune them to maximize the score.

## Full benchmark

After reviewing the pilot, preserve its six answers and verdicts when extending
to 30 questions:

```bash
.venv/bin/python -m scripts.evaluate --seed-capture data/eval/pilot_answers.json --output data/eval/captured_answers.json
.venv/bin/python -m scripts.evaluate_faithfulness --provider anthropic --input data/eval/captured_answers.json --seed-judgments data/eval/pilot_faithfulness_anthropic.json --output data/eval/faithfulness_anthropic.json
```

Seeding requires matching generator settings for capture and matching model,
prompts, full answer, question, and context for judging. Source artifact hashes
are recorded. Original pilot artifacts remain unchanged. Include the same
`--seed-judgments` argument when resuming this full judge run.

Alternatively, generate and judge everything afresh with Gemini:

```bash
.venv/bin/python -m scripts.evaluate --output data/eval/captured_answers.json
.venv/bin/python -m scripts.evaluate_faithfulness --input data/eval/captured_answers.json --output data/eval/faithfulness.json
```

Add `--resume` (and `--provider anthropic` for Claude) to retry errors and continue judging the same captured file. The
input hash, judge model, and prompts must match. Successful verdicts are retained.
The JSON stores raw judge text, returned model versions, token usage, and evidence.
Data artifacts are excluded by the repository's existing `data/` ignore rule.

Each answer scores supported claims / total factual claims. The headline score
is the unweighted mean across scored answers, on a 0–1 scale. Abstentions, empty
claim lists, and judge failures are excluded and counted separately. Always report
the scored denominator, total claims, and exclusions. Do not describe an incomplete
run as a 30-answer result or an unreviewed judge run as human-validated.

## Offline checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```
