# DeepSeek 微调完整演示

这个项目用 **DeepSeek-R1-Distill-Qwen-1.5B** 演示一个可复现的监督微调流程：校验数据 → 评估基础模型 → LoRA 微调 → 评估微调模型 → 单条推理。任务是把中文客服消息分类为 `退款`、`物流`、`发票`。选择分类任务，是为了能用留出的测试集客观比较微调前后的结果；这份小数据集主要用于学习流程，不代表生产效果。

本例训练的是 [DeepSeek 的 1.5B 蒸馏模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 的 LoRA adapter，不是 671B 的完整 DeepSeek-R1。运行时会下载基础模型权重；本仓库不包含权重。请查看该模型页面及其上游模型的许可说明。

## 环境

- Python 3.10–3.12，建议使用独立虚拟环境。
- NVIDIA CUDA GPU 或 Apple Silicon MPS 用于训练。MPS 路径使用 float32，可能需要较多统一内存；建议有充足内存的机器。CPU 可以做数据检查和推理，但本例不在 CPU 上训练。
- 首次运行需要访问 Hugging Face 下载模型。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果机器上只有其他受支持的 Python 版本，把 `python3.11` 换成对应命令。库版本会持续变化；遇到依赖兼容问题时，优先核对 [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer) 和 [PEFT LoRA 文档](https://huggingface.co/docs/peft/main/package_reference/lora)。

若权重下载受网络限制，可先通过官方模型仓库把模型下载到本地目录，再在 `train`、`evaluate`、`predict` 命令中统一传入 `--model /你的/本地模型目录`。三个命令必须使用同一基础模型；只下载配置和 tokenizer 并不足以训练。

## 一次走完整个流程

在项目根目录执行：

```bash
# 1. 检查 JSONL 数据及训练/评估集是否重复；这一步不用下载模型
python deepseek_demo.py check-data

# 2. 评估基础模型，保存逐题原始输出
python deepseek_demo.py evaluate --results artifacts/before.json

# 3. 训练并保存 LoRA adapter
python deepseek_demo.py train

# 4. 用相同测试集评估微调结果
python deepseek_demo.py evaluate \
  --adapter artifacts/deepseek-intent-lora \
  --results artifacts/after.json

# 5. 试一条新消息
python deepseek_demo.py predict \
  --adapter artifacts/deepseek-intent-lora \
  '我的包裹到哪儿了？'
```

比较 `artifacts/before.json` 与 `artifacts/after.json` 中的 `score` 和每道题的 `raw_output`。脚本只把明确出现的 `<answer>退款</answer>`、`<answer>物流</answer>` 或 `<answer>发票</answer>` 视为有效标签；未按格式输出会记为错误。对于推理模型，原始输出可能包含较长的思考文本，所以评估时同时保存完整输出供检查。

## 数据格式

`data/train.jsonl` 每行包含 `prompt`、`completion`、`label`：

```json
{"prompt":[{"role":"user","content":"请判断……客户消息：商品有破损，我要申请退钱。"}],"label":"退款","completion":[{"role":"assistant","content":"<answer>退款</answer>"}]}
```

`data/eval.jsonl` 只包含 `prompt` 和 `label`，不向模型提供正确回答。训练时使用 TRL 的 **conversational prompt-completion** 格式，并只在 completion 上计算损失。真实项目应增加高质量数据、保持类别平衡，并检查相似问题没有同时落入训练集和测试集。

可以用参数调整模型、训练轮数和最大长度，例如：

```bash
python deepseek_demo.py train --epochs 3 --max-length 768 \
  --train-data data/train.jsonl --adapter artifacts/my-adapter
```

训练输出的 adapter **不能单独推理**，加载时仍需同一个基础模型及其 tokenizer。`artifacts/` 被 Git 忽略，避免意外提交权重与生成结果。

## 验证

不下载模型即可运行数据和评估逻辑的测试：

```bash
python -m unittest discover -s tests -v
```

完成一次真实训练后，还应检查 `before.json` 与 `after.json` 的逐题结果；24 条训练样本只够演示流程，准确率可能有波动，也不能据此判断模型的通用能力。DeepSeek 官方还建议对推理任务做多次评估；本例的确定性解码用于便于复现分类对比。[DeepSeek 使用建议](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations)
