"""A small, reproducible DeepSeek intent-classification fine-tuning demo."""

import argparse
import json
import re
from pathlib import Path


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LABELS = {"退款", "物流", "发票"}
ANSWER_RE = re.compile(r"<answer>\s*(退款|物流|发票)\s*</answer>")
ROOT = Path(__file__).resolve().parent


def extract_label(text: str) -> str | None:
    """Read the last explicitly tagged valid answer from a model response."""
    matches = ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def load_examples(path: Path, require_completion: bool) -> list[dict]:
    """Validate local JSONL before sending it to the trainer or evaluator."""
    examples = []
    seen_prompts = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = row["prompt"]
                label = row["label"]
                if label not in LABELS:
                    raise ValueError("unknown label")
                if not isinstance(prompt, list) or len(prompt) != 1:
                    raise ValueError("expected one user message")
                if prompt[0]["role"] != "user":
                    raise ValueError("expected user role")
                content = prompt[0]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty prompt")
                if content in seen_prompts:
                    raise ValueError("duplicate prompt")
                if require_completion:
                    completion = row["completion"]
                    if not isinstance(completion, list) or len(completion) != 1:
                        raise ValueError("expected one answer")
                    if completion[0]["role"] != "assistant":
                        raise ValueError("expected assistant role")
                    if completion[0]["content"] != f"<answer>{label}</answer>":
                        raise ValueError("answer differs from label")
                elif "completion" in row:
                    raise ValueError("evaluation data must not contain answers")
                seen_prompts.add(content)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}: line {line_number} is invalid") from error
            examples.append(row)
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def score_predictions(examples: list[dict], outputs: list[str]) -> dict:
    if len(examples) != len(outputs):
        raise ValueError("Number of outputs must match number of examples")
    correct = sum(extract_label(output) == row["label"]
                  for row, output in zip(examples, outputs))
    return {"correct": correct, "total": len(examples),
            "accuracy": correct / len(examples)}


def runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("请先安装依赖：python -m pip install -r requirements.txt") from error
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    return torch, AutoModelForCausalLM, AutoTokenizer, device, dtype


def load_for_inference(model_id: str, adapter: Path | None):
    torch, model_class, tokenizer_class, device, dtype = runtime()
    tokenizer = tokenizer_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def generate(torch, tokenizer, model, device: str, prompt: list[dict],
             max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def train(args):
    torch, model_class, tokenizer_class, device, dtype = runtime()
    if device == "cpu":
        raise SystemExit("训练需要 NVIDIA CUDA GPU 或 Apple Silicon MPS。")
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = load_examples(args.train_data, require_completion=True)
    tokenizer = tokenizer_class.from_pretrained(args.model)
    model = model_class.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    training_data = Dataset.from_list([
        {"prompt": row["prompt"], "completion": row["completion"]}
        for row in rows
    ])
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=training_data,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16,
            lora_dropout=0.05, target_modules="all-linear",
        ),
        args=SFTConfig(
            output_dir=str(args.adapter),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_length=args.max_length,
            completion_only_loss=True,
            gradient_checkpointing=True,
            bf16=device == "cuda" and dtype == torch.bfloat16,
            fp16=device == "cuda" and dtype == torch.float16,
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.adapter))
    tokenizer.save_pretrained(str(args.adapter))
    print(f"LoRA adapter 已保存：{args.adapter}")


def evaluate(args):
    rows = load_examples(args.eval_data, require_completion=False)
    torch, tokenizer, model, device = load_for_inference(args.model, args.adapter)
    outputs = [generate(torch, tokenizer, model, device, row["prompt"],
                        args.max_new_tokens) for row in rows]
    score = score_predictions(rows, outputs)
    results = [
        {"prompt": row["prompt"][0]["content"], "expected": row["label"],
         "predicted": extract_label(output), "raw_output": output}
        for row, output in zip(rows, outputs)
    ]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({"score": score, "examples": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"准确率：{score['correct']}/{score['total']} = {score['accuracy']:.1%}")
    print(f"逐题结果：{args.results}")


def predict(args):
    prompt = [{"role": "user", "content":
               f"请判断这条客户消息的意图，只能选择退款、物流、发票之一。"
               f"最终答案写成 <answer>标签</answer>。客户消息：{args.text}"}]
    torch, tokenizer, model, device = load_for_inference(args.model, args.adapter)
    output = generate(torch, tokenizer, model, device, prompt, args.max_new_tokens)
    print(output)
    print("识别标签：", extract_label(output) or "未识别")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-data", help="不下载模型，检查示例数据")
    train_parser = sub.add_parser("train", help="训练 LoRA adapter")
    eval_parser = sub.add_parser("evaluate", help="评估基础模型或 adapter")
    predict_parser = sub.add_parser("predict", help="对一条消息推理")
    for child in (train_parser, eval_parser, predict_parser):
        child.add_argument("--model", default=MODEL_ID)
        child.add_argument("--adapter", type=Path, default=None)
    train_parser.set_defaults(adapter=ROOT / "artifacts/deepseek-intent-lora")
    train_parser.add_argument("--train-data", type=Path, default=ROOT / "data/train.jsonl")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--max-length", type=int, default=512)
    eval_parser.add_argument("--eval-data", type=Path, default=ROOT / "data/eval.jsonl")
    eval_parser.add_argument("--results", type=Path, default=ROOT / "artifacts/evaluation.json")
    eval_parser.add_argument("--max-new-tokens", type=int, default=256)
    predict_parser.add_argument("text")
    predict_parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.command == "check-data":
        train_rows = load_examples(ROOT / "data/train.jsonl", True)
        eval_rows = load_examples(ROOT / "data/eval.jsonl", False)
        train_prompts = {row["prompt"][0]["content"] for row in train_rows}
        eval_prompts = {row["prompt"][0]["content"] for row in eval_rows}
        if train_prompts & eval_prompts:
            raise SystemExit("训练集与评估集存在重复问题")
        print(f"训练样本：{len(train_rows)}；评估样本：{len(eval_rows)}")
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
