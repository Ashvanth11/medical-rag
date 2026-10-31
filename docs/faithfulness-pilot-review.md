# Six-answer faithfulness pilot review

This is an assistant inspection of the saved benchmark artifacts, not a human
review or validation of medical correctness. No clinical conclusions are made.

Claude Sonnet 5.5 judged six captured Llama 3.1 answers. The unweighted mean
per-answer score was **0.898148 (89.8%)**. Across scored answers there were
46 supported labels out of 48 claims; this pooled fraction is a different metric.
There were no abstentions or remaining judge errors after evidence revalidation.
This is a six-answer pilot, not the full 30-question benchmark.

| Answer ID | Supported / claims | Score | Inspection |
| --- | --- | --- | --- |
| hbp_001 | 9 / 9 | 100% | Quotes match the context. Some equivalent prevalence assertions are counted separately, while the two leading-cause assertions are grouped. Claim granularity needs human review. |
| diabetes_001 | 1 / 2 | 50% | The second verdict is disputed: the context places an unlabeled fasting-test sentence directly after the A1C sentence. The judge invokes a fasting plasma glucose interpretation that is not explicitly labeled in the captured chunk. Treat this as ambiguous context plus a possible outside-knowledge judgment, not confirmed unsupported generation. |
| cholesterol_001 | 11 / 11 | 100% | Evidence supports the extracted assertions. Repeated or near-equivalent assertions affect claim counts. |
| obesity_001 | 1 / 1 | 100% | The threshold appears in the chunk. The answer includes a source dictionary in its output; this is a presentation problem the faithfulness score does not measure. The source's adult-population qualifier also warrants review. |
| stroke_001 | 16 / 16 | 100% | Evidence quotes match the captured context after decoding two double-escaped Unicode em dashes. The original judge responses remain saved. |
| copd_001 | 8 / 9 | 88.9% | The unsupported verdict's reason contradicts its label: it acknowledges the wording matches garbled source text. The chunk contains `before treatment goals. are recognized.` This needs adjudication; the score must not be presented as established hallucination detection. |

The two negative labels are the most consequential calibration issues. A faithful
answer can repeat corrupted source text. Faithfulness must not quietly become a
medical correctness test that relies on the judge's outside knowledge. A human
should review these cases and the extraction granularity before freezing prompts
and running the full benchmark. Do not overwrite the raw scores with subjective
corrections; retain any adjudications separately.

## Artifacts and execution

- Capture: `data/eval/pilot_answers.json`
- Claude verdicts and raw responses: `data/eval/pilot_faithfulness_anthropic.json`
- Readable claims and complete excerpts: `data/eval/pilot_faithfulness_anthropic.md`
- Original result before Unicode recovery: `data/eval/pilot_faithfulness_anthropic_before_unicode_fix.json`
- Gemini was stopped; its one completed result remains in `data/eval/pilot_faithfulness.json`.

Claude request time totaled 51.90 seconds across 13 successful responses, including
the stroke quote-validation retry. Recorded usage was 28,634 input and 7,439 output
tokens. At the checked standard rates of $2 / million input and $10 / million
output, estimated token cost is $0.132; this is not an account billing statement.
Pricing source: https://platform.claude.com/docs/en/models/sonnet-5-5/overview

No extra API calls were made to recover the stroke result: the final saved raw
response was revalidated with deterministic Unicode escape decoding, and only
quotes that exactly matched the original chunk text were accepted. Both the raw
quote and resolved quote are retained. Six offline tests pass.
