import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deepseek_demo import extract_label, load_examples, score_predictions


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_rejects_wrong_answer_for_training_example(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({
                "prompt": [{"role": "user", "content": "我要退货"}],
                "completion": [{"role": "assistant", "content": "<answer>发票</answer>"}],
                "label": "退款",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_examples(path, require_completion=True)

    def test_rejects_answer_leakage_in_evaluation_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_eval.jsonl"
            path.write_text(json.dumps({
                "prompt": [{"role": "user", "content": "物流到哪了"}],
                "label": "物流",
                "completion": [{"role": "assistant", "content": "<answer>物流</answer>"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_examples(path, require_completion=False)

    def test_extracts_only_explicit_final_answer(self):
        self.assertEqual(extract_label("先判断意图。<answer>退款</answer>"), "退款")
        self.assertIsNone(extract_label("可能是退款或物流"))
        self.assertIsNone(extract_label("<answer>未知</answer>"))

    def test_scores_invalid_output_as_incorrect(self):
        examples = [{"label": "退款"}, {"label": "物流"}]
        result = score_predictions(examples, ["<answer>退款</answer>", "无法判断"])
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 0.5)

    def test_example_files_are_valid_and_do_not_overlap(self):
        train = load_examples(ROOT / "data/train.jsonl", require_completion=True)
        eval_set = load_examples(ROOT / "data/eval.jsonl", require_completion=False)
        self.assertGreaterEqual(len(train), 20)
        self.assertGreaterEqual(len(eval_set), 6)
        self.assertFalse({row["prompt"][0]["content"] for row in train} &
                         {row["prompt"][0]["content"] for row in eval_set})

    def test_check_data_runs_without_training_dependencies(self):
        result = subprocess.run(
            [sys.executable, "deepseek_demo.py", "check-data"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("训练样本", result.stdout)


if __name__ == "__main__":
    unittest.main()
