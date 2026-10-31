# Faithfulness benchmark: October 31, 2026

**Claude Sonnet 5.5-judged faithfulness: 89.9% across 30 answers.**

This is the unweighted mean of each answer's supported-claim fraction. Claude
labeled 315 of 344 extracted claims supported and 29 unsupported. The pooled
claim fraction is 91.6%; it is not the headline metric. All 30 answers scored,
with zero abstentions, zero empty claim lists, and zero remaining judge failures.
Twenty answers scored 100%; ten scored below 100%.

## Method and scope

- Generator: the project's local Ollama `llama3.1` configuration, temperature 0,
  top-k 5 before URL deduplication. Six pilot answers were retained; the other
  24 were captured in the full run.
- Judge: `claude-sonnet-5-5`, two stages for claim extraction and support judgment,
  structured JSON, medium effort, `between_tools` thinking, 4,096 output-token cap.
- Evidence: all exact post-deduplication context chunks supplied to the generator,
  not just cited sources. Full answers, prompts, context, verdicts, evidence quotes,
  usage, and returned judge model IDs are saved.
- Six pilot verdicts were reviewed by the user. The user's decision was to retain
  the original labels and caveats. The remaining answers have not all been human
  adjudicated. Scores are reported without subjective manual corrections.
- Supported claims require evidence that exactly matches a chunk. Double-escaped
  Unicode is decoded only when the decoded quote matches exactly; raw quotes are
  retained. Invalid evidence caused retries, not automatic unsupported labels.
- These are judge-model estimates of context support, not medical accuracy,
  completeness, citation correctness, or clinical safety. A faithful response can
  repeat corrupted source text. Repeated assertions and claim segmentation also
  affect the counts.

## Per-answer results

| ID | Supported / claims | Faithfulness |
| --- | --- | --- |
| hbp_001 | 9 / 9 | 100.0% |
| hbp_002 | 8 / 15 | 53.3% |
| hbp_003 | 12 / 19 | 63.2% |
| diabetes_001 | 1 / 2 | 50.0% |
| diabetes_002 | 11 / 11 | 100.0% |
| diabetes_003 | 0 / 1 | 0.0% |
| diabetes_004 | 3 / 3 | 100.0% |
| diabetes_005 | 22 / 22 | 100.0% |
| cholesterol_001 | 11 / 11 | 100.0% |
| cholesterol_002 | 15 / 15 | 100.0% |
| cholesterol_003 | 10 / 10 | 100.0% |
| obesity_001 | 1 / 1 | 100.0% |
| obesity_002 | 2 / 2 | 100.0% |
| obesity_003 | 15 / 15 | 100.0% |
| stroke_001 | 16 / 16 | 100.0% |
| stroke_002 | 25 / 25 | 100.0% |
| copd_001 | 8 / 9 | 88.9% |
| copd_002 | 20 / 21 | 95.2% |
| copd_003 | 17 / 17 | 100.0% |
| arthritis_001 | 7 / 7 | 100.0% |
| arthritis_002 | 4 / 4 | 100.0% |
| kidney_001 | 2 / 2 | 100.0% |
| kidney_002 | 5 / 5 | 100.0% |
| alzheimers_001 | 17 / 18 | 94.4% |
| alzheimers_002 | 13 / 15 | 86.7% |
| tobacco_001 | 2 / 2 | 100.0% |
| tobacco_002 | 16 / 16 | 100.0% |
| alcohol_001 | 3 / 3 | 100.0% |
| heart_001 | 15 / 22 | 68.2% |
| heart_002 | 25 / 26 | 96.2% |

## Retained caveats

The [pilot inspection](faithfulness-pilot-review.md) records ambiguous A1C context,
garbled COPD source text, and a COPD verdict whose rationale contradicts its label.
These remain in the score as requested.

Additional full-run inspection found similar issues. For example, `hbp_003` has an
unsupported physical-activity verdict whose reason ends by acknowledging reasonable
support. Some `hbp_002` reasons do not consistently combine evidence across chunks.
Several `heart_001` judgments reject indirect prevention inferences. These examples
show that strictness and internal consistency vary; the 29 unsupported labels must
not be presented as 29 independently verified hallucinations.

The `diabetes_003` score comes from one extracted claim: the answer says "more than
115 million," while the judge reads the context as "about 115 million." A zero
score for this answer is therefore a single-qualifier judgment, not a broad test
of the answer's medical usefulness.

## Artifacts, cost, and checks

- Full capture: `data/eval/captured_answers.json`
- Verdict JSON and raw responses: `data/eval/faithfulness_anthropic.json`
- Readable claims and full context: `data/eval/faithfulness_anthropic.md`
- Workflow and commands: [faithfulness.md](faithfulness.md)

Recorded Anthropic usage, including the reused pilot and validation retries, was
148,883 input tokens and 50,125 output tokens across 65 responses. At the checked
standard rates of $2 / million input and $10 / million output tokens, estimated
cost is **$0.80 total**, including approximately $0.13 for the pilot. This is not
an account billing statement and excludes the interrupted Gemini attempt.
[Pricing source](https://platform.claude.com/docs/en/models/sonnet-5-5/overview).

Recorded Anthropic request time totaled 318.86 seconds, including pilot calls and
retries; this excludes local answer generation and waiting between requests.
The six pilot judgments were reused without API calls. All evidence quotes, score
arithmetic, question IDs, and unchanged pilot labels were checked locally. Seven
offline tests pass. Implementation file hashes and capture/seed hashes are recorded
with the results. Local evaluation artifacts remain excluded by the `data/` Git
ignore rule; keys stay in the ignored `.env` file.
