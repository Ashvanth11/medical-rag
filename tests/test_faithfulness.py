import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from scripts.evaluate_faithfulness import (
    CLAIMS_SCHEMA, JudgeHTTPError, aggregate, anthropic_generate,
    seed_judgments, strict_schema, validate_claims, validate_verdicts,
)


class ScoringTests(unittest.TestCase):
    def test_seed_verdicts_require_same_answer_evidence_and_judge(self):
        metadata = {"judge_model": "judge", "provider": "anthropic"}
        chunk = {"text": "fact"}
        verdict = {"claim_index": 0, "supported": True, "reason": "Matches",
                   "evidence": [{"chunk_id": 1, "quote": "fact"}]}
        result = dict(id="q1", question="q", answer="fact", status="scored",
                      claims=["fact"], verdicts=[verdict], context_chunks=[chunk], faithfulness=1)
        seed = {"metadata": metadata, "results": [result]}
        row = dict(id="q1", question="q", answer="fact", trace={"status": "answered", "context_chunks": [chunk]})
        self.assertEqual(len(seed_judgments(seed, [row], metadata)), 1)
        with self.assertRaises(ValueError):
            seed_judgments(seed, [dict(row, answer="different")], metadata)
        with self.assertRaises(ValueError):
            seed_judgments(seed, [dict(row, trace={"status": "answered", "context_chunks": [{"text": "other"}]})], metadata)
        with self.assertRaises(ValueError):
            seed_judgments(seed, [row], dict(metadata, judge_model="other"))

    def test_anthropic_request_and_response(self):
        session = Mock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {
            "content": [{"type": "text", "text": '{"claims":["fact"]}'}],
            "model": "claude-sonnet-5-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        result = anthropic_generate(session, "test-key", "claude-sonnet-5-5", "extract", {}, CLAIMS_SCHEMA, 45)
        self.assertEqual(result["model_version"], "claude-sonnet-5-5")
        self.assertEqual(result["usage"]["input_tokens"], 100)
        request = session.post.call_args.kwargs
        self.assertEqual(request["timeout"], (10, 45))
        self.assertEqual(request["json"]["thinking"], {"type": "between_tools"})
        self.assertNotIn("temperature", request["json"])
        self.assertFalse(request["json"]["output_config"]["format"]["schema"]["additionalProperties"])
        self.assertNotIn("additionalProperties", CLAIMS_SCHEMA)
        session.post.return_value.status_code = 401
        with self.assertRaises(JudgeHTTPError) as error:
            anthropic_generate(session, "test-key", "claude-sonnet-5-5", "extract", {}, CLAIMS_SCHEMA, 45)
        self.assertEqual(error.exception.code, 401)

    def test_macro_average_and_explicit_exclusions(self):
        result = aggregate([
            dict(status="scored", supported_claims=1, total_claims=1, faithfulness=1),
            dict(status="scored", supported_claims=1, total_claims=4, faithfulness=.25),
            dict(status="abstained"), dict(status="no_claims"), dict(status="error"),
        ])
        self.assertEqual(result["faithfulness"], .625)
        self.assertEqual(result["total_claims"], 5)
        self.assertEqual(result["n_scored"], 2)
        self.assertIsNone(aggregate([])["faithfulness"])

    def test_evidence_validation(self):
        chunks = [{"text": "A supported fact."}]
        good = {"claim_index": 0, "supported": True, "reason": "Entailed",
                "evidence": [{"chunk_id": 1, "quote": "supported fact"}]}
        self.assertEqual(validate_verdicts({"verdicts": [good]}, ["fact"], chunks), [good])
        for evidence in ([], [{"chunk_id": 2, "quote": "fact"}],
                         [{"chunk_id": 1, "quote": "invented"}]):
            with self.assertRaises(ValueError):
                validate_verdicts({"verdicts": [dict(good, evidence=evidence)]}, ["fact"], chunks)
        with self.assertRaises(ValueError):
            validate_verdicts({"verdicts": []}, ["fact"], chunks)
        with self.assertRaises(ValueError):
            validate_claims({"claims": ["x", "x"]})

    def test_double_escaped_unicode_requires_exact_match(self):
        quote = r"short time\u2014usually"
        verdict = {"claim_index": 0, "supported": True, "reason": "Entailed",
                   "evidence": [{"chunk_id": 1, "quote": quote}]}
        result = validate_verdicts({"verdicts": [verdict]}, ["fact"], [{"text": "short time—usually"}])
        self.assertEqual(result[0]["evidence"][0]["quote"], "short time—usually")
        self.assertEqual(result[0]["evidence"][0]["raw_quote"], quote)
        verdict["evidence"][0]["quote"] = r"invented\u2014usually"
        with self.assertRaises(ValueError):
            validate_verdicts({"verdicts": [verdict]}, ["fact"], [{"text": "short time—usually"}])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        # Load pipeline with test doubles so tests need neither Qdrant nor embeddings.
        retrieve = ModuleType("src.rag.retrieve")
        retrieve.retrieve_top_k = lambda *a, **kw: (
            [{"url": "a", "text": "first"}, {"url": "a", "text": "duplicate"},
             {"url": "b", "text": "uncited context"}], [.9, .8, .7])
        classify = ModuleType("src.rag.classify")
        classify.QueryClass = type("QueryClass", (), {"OFF_TOPIC": "off"})
        classify.classify_query = lambda q: "off"
        spec = importlib.util.spec_from_file_location("test_pipeline", Path("src/rag/pipeline.py"))
        self.pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"src.rag.retrieve": retrieve, "src.rag.classify": classify}):
            spec.loader.exec_module(self.pipeline)

    def test_full_context_differs_from_cited_sources(self):
        trace = {}
        with patch.object(self.pipeline, "generate_answer", return_value="first [1]"):
            answer, sources, history = self.pipeline.answer_question("q", trace=trace)
        self.assertEqual(len(sources), 1)
        self.assertEqual([c["text"] for c in trace["context_chunks"]], ["first", "uncited context"])
        self.assertIn("uncited context", trace["user_prompt"])
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(history[-1]["content"], answer)

    def test_new_fallback_is_abstention(self):
        trace = {}
        with patch.object(self.pipeline, "generate_answer", return_value="No citation"):
            self.pipeline.answer_question("q", trace=trace)
        self.assertEqual(trace["status"], "abstained")


if __name__ == "__main__":
    unittest.main()
