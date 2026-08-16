# 基于证据感知多路径推理的目标级多模态隐式情感分析

本项目实现当前实验设计的完整训练与测试流程。Twitter-2015 和 Twitter-2017 **严格分开训练、分开评测**；当前可先将处理完成的 Twitter-2015 数据放入对应目录运行，Twitter-2017 目录和配置已经预留。

## 1. 项目结构

```text
target_multimodal_implicit_sentiment/
├── README.md
├── AGENTS.md
├── .gitattributes
├── .gitignore
├── .github/workflows/validate.yml
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── twitter2015.yaml
│   └── twitter2017.yaml
├── data/                         # 服务器手动放置，整个目录不进入 Git
│   ├── twitter2015/
│   │   ├── README.md
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── images/
│   └── twitter2017/
│       ├── README.md
│       ├── train.json
│       ├── dev.json
│       ├── test.json
│       └── images/
├── tests/fixtures/
│   └── sample_record.json        # 仅用于 smoke test 的合成样例
├── scripts/
│   ├── validate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_training_report.py
│   ├── validate_training_report.py
│   ├── run_server_pipeline.sh
│   ├── push_training_report.sh
│   └── infer_one.py
├── reports/
│   └── README.md
├── src/tmis/
│   ├── constants.py
│   ├── config.py
│   ├── runtime.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── conditioning.py
│   │   ├── evidence.py
│   │   ├── reasoning.py
│   │   ├── bridge.py
│   │   └── model.py
│   ├── training/
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── checkpoint.py
└── outputs/
```

## 2. 模型计算链

```text
restored_text → T5-large Encoder → H_t
                                  │
target ───────→ shared T5 Encoder → h_a (Target Query)
                                  │
image ────────→ CLIP ViT-L/14 ──→ H_v
                 │                │
                 ▼                ▼
       Target-Conditioned Text / Visual Attention
                         │
                    H_t^a, H_v^a
                         │
             H_t^a + H_v^a + h_a
                         │
                         ▼
          Target-Conditioned Multimodal Fusion
                         │
                        H_f
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
       Evidence Heads   Reasoning Tags   Cross-modal Head
         H_te, H_ve       p_e,p_i,p_c        H_relation
                \             |             /
                 \            |            /
                   Three Reasoning Paths
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
 Direct Reasoning  Implicit Reasoning  Cross-modal Reasoning
       └───────────────┼────────────────┘
                       ▼
              Cross-Path Interaction
                       │
                  H_reasoning
                       │
              T5-large Bridge Decoder
 [GROUND] → [TRANSITION] → [IMPLICATION]
                       │
                       ▼
               Bridge Encoder only
                       │
                       ▼
              Sentiment Classifier
```

文本、目标和辅助 evidence 文本复用同一个 T5-large encoder；reasoning bridge 复用同一模型的预训练 T5 decoder、共享词嵌入和 LM head。项目不会额外加载第二套 T5-large。图像编码器为 `openai/clip-vit-large-patch14`，T5/CLIP 输出再投影到配置中的统一多模态维度。

Bridge 使用自定义 `<BRIDGE_BOS>` 与三个字段 marker，但结束符复用 T5 原生 `</s>`，避免重新学习第二套 EOS 语义。

最终分类器**不读取** `H_f`、证据表示、reasoning tags、路径表示或图像/文本特征，只读取模型生成的 reasoning bridge token 序列。因此最终预测链保持：

```text
多模态输入 → 推理 → reasoning_bridge → sentiment
```

## 3. 数据要求

把 Twitter-2015 数据直接替换到：

```text
data/twitter2015/train.json
data/twitter2015/dev.json
data/twitter2015/test.json
data/twitter2015/images/
```

JSON 根节点可以是：

- 直接的 `[...]`
- `{"data": [...]}`
- `{"records": [...]}`

目标字段兼容现有数据中的 `targe`，也兼容 `target`。

正式训练字段：

```json
{
  "restored_text": "...",
  "targe": "...",
  "image": "...",
  "sentiment": "positive|neutral|negative",
  "text_evidence": ["..."],
  "visual_evidence": ["..."],
  "reasoning_tags": {
    "explicit_cue_present": false,
    "implicit_reasoning_required": true,
    "cross_modal_reasoning_required": true
  },
  "reasoning_bridge": {
    "grounded_synthesis": "...",
    "reasoning_transition": "...",
    "evaluative_implication": "..."
  }
}
```

其中 `text_evidence` 按当前数据设计必须是 `restored_text` 中的精确连续子串。默认配置会在训练前严格检查。

`reasoning_bridge` 缺失的样本不会从最终测试集删除：

- Stage 1 可继续用于 evidence/tags 辅助训练；
- Stage 2–4 中需要 reference bridge 的训练步骤只选择存在 bridge 的训练样本；
- Stage 3 的最佳 Bridge checkpoint 需要开发集中至少存在一条 reference bridge，否则无法计算选优指标，训练器会明确报错；
- Stage 5 与最终测试使用模型自己生成的 bridge，因此最终 sentiment evaluation 仍可覆盖完整测试集。

## 4. 安装

建议 Python 3.10+：

```bash
git clone <repository-url>
cd target_multimodal_implicit_sentiment
pip install -r requirements.txt
pip install -e .
```

本仓库不保存或同步任何训练数据，整个 `data/` 目录已被 `.gitignore` 永久排除。服务器 clone 代码后，请手动把数据 JSON 和图片放到配置所指向的 `data/twitter2015/`、`data/twitter2017/` 目录；后续执行 `git add` 或训练报告推送脚本也不会把这些数据提交到 GitHub。数据放置完成后运行：

```bash
python scripts/validate_data.py --config configs/twitter2015.yaml
python scripts/validate_data.py --config configs/twitter2017.yaml
```

首次运行会通过 Hugging Face 下载 `google-t5/t5-large` 与 `openai/clip-vit-large-patch14`，配置已固定两个模型仓库 revision 并优先加载 safetensors，避免服务器在不同日期静默得到不同权重。离线服务器可先缓存模型，再设置 `model.local_files_only: true`。该组合远大于旧的 BERT-base + CLIP ViT-B/32；默认配置已将单卡 batch 调整为 1、梯度累积调整为 16，开启两个 backbone 的 gradient checkpointing，并使用 Adafactor 降低优化器状态显存。

本架构与旧 BERT/CLIP-B/32 checkpoint 的参数名、隐藏维度和 tokenizer 均不兼容，必须从新的 T5-large/CLIP-L/14 预训练权重重新开始训练。

## 5. 训练前检查

先放入 Twitter-2015 数据和图片，然后执行：

```bash
python scripts/validate_data.py --config configs/twitter2015.yaml
```

检查内容包括：

- JSON 结构；
- sentiment 值；
- target/restored_text/image 是否存在；
- 图片文件是否可解析；
- `text_evidence` 是否为精确连续子串；
- reasoning tags；
- reasoning bridge 三字段结构；
- positive / neutral / negative 数量；
- implicit / non-implicit 数量；
- bridge 缺失数量。

隐式子集仅由：

```text
reasoning_tags.implicit_reasoning_required == true
```

确定。`explicit_cue_present` 与 `cross_modal_reasoning_required` 不参与隐式/非隐式定义。

## 6. 五阶段训练

运行：

```bash
python scripts/train.py --config configs/twitter2015.yaml
```

单机多卡使用 PyTorch DDP（每张 GPU 一个进程）：

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config configs/twitter2015.yaml
```

训练器会在每轮结束后原子更新 `outputs/twitter2015/latest.pt`，其中包含模型、优化器、学习率调度器和 AMP scaler 状态。服务器中断后可继续：

```bash
python scripts/train.py \
  --config configs/twitter2015.yaml \
  --resume outputs/twitter2015/latest.pt
```

默认混合精度为 `auto`：支持 BF16 的 CUDA GPU 使用 BF16，否则 CUDA 使用 FP16，CPU 回退 FP32。若显存仍不足，可以在某一阶段设置 `train_text_backbone: false` 或 `train_vision_backbone: false`，只训练该阶段的新模块。

每次正式实验应指定唯一 `run-id` 和独立输出目录，避免旧实验文件混入新报告：

```bash
python scripts/train.py \
  --config configs/twitter2015.yaml \
  --run-id twitter2015-run-001 \
  --output-dir outputs/twitter2015/runs/twitter2015-run-001
```

训练入口会自动写入 `run_state.json`，记录训练状态、代码 commit、分支、工作区状态、Python/PyTorch/Transformers/CUDA 版本和 GPU 信息。API key、GitHub token 与服务器主机名不会写入该文件。

程序按顺序执行：

### Stage 1 — 辅助监督预训练

训练目标条件编码、多模态融合、文本证据头、视觉证据头和 reasoning tags head。

```text
L_stage1 = λ_te L_text_evidence
         + λ_ve L_visual_evidence
         + λ_tag L_reasoning_tags
```

- Text Evidence：token-level BCE；标签由精确 `text_evidence` 子串自动映射得到；
- Visual Evidence：不使用 bounding box，使用目标条件视觉证据表示与已验证 `visual_evidence` 文本表示之间的语义对齐/对比损失；
- Reasoning Tags：三个独立 sigmoid + BCE。

### Stage 2 — 多路径推理 warm-up

三条路径全部运行，reasoning tags 仅产生软路由权重：

```text
r_k = epsilon + alpha_k * p_k
```

训练初期支持：

```text
p_route = λ * gold_tag + (1-λ) * predicted_probability
```

随后 λ 逐步降低。为了使三条推理路径在没有额外 path-level 人工标签的前提下得到实际语义训练，本实现用 reference reasoning bridge 的生成损失作为 reasoning-path warm-up 的监督信号，同时保留 tag loss；不会增加一个可绕过 bridge 的上游 sentiment head。

### Stage 3 — Reasoning Bridge Generator

将 reference bridge 序列化为：

```text
[GROUND] grounded_synthesis
[TRANSITION] reasoning_transition
[IMPLICATION] evaluative_implication
```

采用与文本编码器同源的预训练 T5-large decoder，通过 teacher forcing 学习。五个多模态 memory token 会先从融合维度投影到 T5 `d_model`，再作为 `encoder_outputs` 供 T5 decoder 交叉注意力读取：

```text
P(S,R,E) = P(S) P(R|S) P(E|S,R)
```

每个 Stage 3 epoch 结束后，开发集会关闭 gold routing，并由模型真实自回归生成完整 Bridge。程序记录结构合法率与 reference ROUGE-L，默认按完整 Bridge 的 ROUGE-L F1 保存 `best_bridge.pt`；Stage 3 结束后会自动恢复该最佳权重，再进入 Stage 4。Reference Bridge 在这里仅作为评价目标，不作为 decoder 输入。

### Stage 4 — Bridge Encoder + Sentiment Classifier

只使用 reference reasoning bridge 训练最终分类器：

```text
reasoning_bridge → Bridge Encoder → positive/neutral/negative
```

此阶段分类器没有任何从多模态表示直接进入的旁路。

### Stage 5 — 系统级联合微调

联合优化：

```text
L_total = λ1 L_text_evidence
        + λ2 L_visual_evidence
        + λ3 L_tags
        + λ4 L_bridge
        + λ5 L_sentiment
```

分类器训练逐步混入模型生成 bridge。由于生成 token 是离散序列，`L_sentiment` **不会伪装成能够普通反向传播穿过 argmax 生成过程**：

- sentiment loss 更新 Bridge Encoder / classifier；
- 上游多模态与推理模块由 evidence/tag/bridge generation losses 更新；
- 这与真实推理时“先生成 bridge，再分类”的计算路径一致。

默认 Stage 5 使用四个 epoch，将 Generated Bridge 比例依次调整为 `25% → 50% → 75% → 100%`。训练器会强制最后一轮为 100%，并额外保存 `stage5_generated_only.pt`；`best_joint.pt` 仍按开发集 Macro-F1 选择，两者含义不会混淆。

所有阶段参数、学习率、epoch、loss 权重及 generated/reference bridge 混合比例均在 YAML 中配置。

## 7. 测试

训练完成后：

```bash
python scripts/evaluate.py \
  --config configs/twitter2015.yaml \
  --checkpoint outputs/twitter2015/best_joint.pt
```

输出：

```text
outputs/twitter2015/test_metrics.json
outputs/twitter2015/test_predictions.json
```

指标自动分成：

- `full`
- `implicit`
- `non_implicit`

三个子集，并同时报告 Accuracy、Macro-Precision、Macro-Recall、Macro-F1。

测试时只输入：

```text
restored_text + image + target
```

不会把 gold `text_evidence`、`visual_evidence`、`reasoning_tags` 或 `reasoning_bridge` 输入模型推理路径。证据表示和 reasoning tags 都由模型预测。

## 8. 单样本推理

```bash
python scripts/infer_one.py \
  --config configs/twitter2015.yaml \
  --checkpoint outputs/twitter2015/best_joint.pt \
  --text "Chuck Bass is everything # MCM" \
  --target "Chuck Bass" \
  --image /path/to/image.jpg
```

对外结果只有：

```json
{
  "reasoning_bridge": "grounded synthesis ... reasoning transition ... evaluative implication ...",
  "sentiment": "positive"
}
```

内部三段 bridge 保持固定标记和顺序；对人展示时仅对同一生成结果进行确定性拼接，不调用第二个模型重新生成解释。

## 9. Twitter-2017

Twitter-2017 已预留独立数据目录和独立配置：

```text
data/twitter2017/
configs/twitter2017.yaml
```

后续只需放入 Twitter-2017 自己的 train/dev/test 和图片，然后分别运行：

```bash
python scripts/validate_data.py --config configs/twitter2017.yaml
python scripts/train.py --config configs/twitter2017.yaml
python scripts/evaluate.py --config configs/twitter2017.yaml
```

不要将 Twitter-2015 与 Twitter-2017 直接合并训练；两个数据集的 checkpoint 和 outputs 也保持独立。

## 10. V2 训练与评测完整性约束

本版本额外固定以下实验约束：

- `reasoning_tags` 三个字段必须是真正 JSON boolean，字符串 `"false"` 不会被错误转换成 `True`。
- `text_evidence` 保留原始内部空格并按 `restored_text` 精确连续子串检查；若证据在 `max_text_length` 截断后完全不可见，则该样本的 text-evidence token loss 会被忽略，而不是错误地作为“全负样本”训练。
- `visual_evidence` 文本只在需要计算视觉证据对比损失时编码；正常 dev/test sentiment 推理显式关闭这一监督分支。
- `routing_gold_mix=0` 时不会把 gold reasoning tags 传入 Soft Router。
- reference bridge 使用字段感知截断，始终保留 `[GROUND]`、`[TRANSITION]`、`[IMPLICATION]` 三个结构标记。
- 自回归 bridge 生成带有结构顺序约束，只约束 marker 顺序，不决定字段语义或 sentiment。
- Stage 5 的 generated bridge 通过 `no_grad` 离散生成；`L_sentiment` 不被错误描述为可穿过 argmax token 反传到 Bridge Generator。
- Stage 5 生成 Bridge 时临时切换到 eval 模式，避免 Dropout 造成训练生成分布与开发/测试生成分布不一致。
- Bridge 自回归解码使用 T5 KV cache；五个 memory token 加入可学习类型嵌入，以区分 reasoning/relation/text-evidence/visual-evidence/target。
- 视觉证据损失带 presence 监督与跨 batch 负样本队列，因此单卡 batch size 为 1 时对比学习也不会退化成恒零损失。
- sentiment、reasoning tags 与文本 evidence 分别启用类别重加权/焦点损失，降低长尾标签对训练的影响。
- 文本、图像和目标先各自池化，再由三路门控融合，避免 CLIP patch token 较多导致图像模态天然占优。
- 每个训练阶段每轮保存独立 JSON loss 日志，便于论文绘制训练曲线和排查 loss 比例。

### Bridge 自动评价

完整测试后，`scripts/evaluate.py` 会同时保存：

- sentiment：Full / Implicit / Non-implicit 的 Accuracy、Macro-P/R/F1；
- bridge structure valid rate；
- reference bridge 的 ROUGE-L（整体以及 S/R/E 分字段）。

ROUGE-L 仅作为词面辅助指标，因为自然语言推理不存在唯一表述。需要语义相似度时安装：

```bash
pip install -r requirements-eval.txt
python scripts/evaluate.py --config configs/twitter2015.yaml --bertscore
```

也可以在已经生成 `test_predictions.json` 后离线运行：

```bash
python scripts/evaluate_bridge.py --config configs/twitter2015.yaml --bertscore
```

### LLM Judge（只用于训练完成后的离线辅助评价）

LLM Judge **不在训练 batch 内调用，也不作为可反向传播 loss**。默认 gold-blind：只读取 `restored_text + target + verified evidence + generated bridge`，不把 gold sentiment 提供给 Judge。

```bash
pip install -r requirements-eval.txt
export LLM_JUDGE_API_KEY="..."
export LLM_JUDGE_BASE_URL="你的 OpenAI-compatible base URL"
python scripts/llm_judge_bridge.py \
  --config configs/twitter2015.yaml \
  --model YOUR_JUDGE_MODEL
```

输出评价维度包括 evidence faithfulness、target ownership、reasoning coherence、field-role separation、evaluative clarity 和 hallucination rate。该分数应作为辅助分析，最终任务结论仍以 sentiment Accuracy/Macro-F1、隐式子集表现和消融实验为主。

### 辅助头与路由诊断

```bash
python scripts/evaluate_auxiliary.py \
  --config configs/twitter2015.yaml \
  --checkpoint outputs/twitter2015/best_joint.pt
```

输出 text-evidence token F1、3 个 reasoning-tag 指标、视觉证据正样本 cosine 以及 implicit/non-implicit 子集平均路由权重。该脚本显式使用 gold 辅助标注作为“评价目标”，与最终 sentiment 的 clean inference 分开执行。

### 无需下载预训练模型的代码检查

```bash
python scripts/smoke_test.py
python scripts/audit_architecture.py
```

`audit_architecture.py` 会检查 Bridge Encoder/Classifier 的 forward API 仅接受 bridge token 和 attention mask，以及五阶段训练、soft-routing curriculum、generated-bridge mixing 和主要 loss 是否仍存在。

## 11. GitHub—服务器—GPT 训练闭环

仓库只将源代码、配置和轻量训练报告纳入版本控制；整个 `data/`、`outputs/`、checkpoint 和下载的预训练权重都不会进入 Git。服务器上的数据由使用者手动放置并始终保留在本地。

### 一条命令执行服务器流水线

Linux 服务器安装依赖并手动放置数据后，可运行：

```bash
NPROC_PER_NODE=4 bash scripts/run_server_pipeline.sh \
  configs/twitter2015.yaml \
  twitter2015-run-001
```

流水线依次执行：数据校验、多卡/单卡训练、完整测试、辅助头评测、报告导出和报告校验。每个 run 使用：

```text
outputs/<dataset>/runs/<run-id>/    # 本地 checkpoint 和完整日志，Git 忽略
reports/<dataset>/<run-id>/         # 轻量可审计报告，提交 Git
```

断点恢复时：

```bash
RESUME_CHECKPOINT=outputs/twitter2015/runs/twitter2015-run-001/latest.pt \
NPROC_PER_NODE=4 \
bash scripts/run_server_pipeline.sh configs/twitter2015.yaml twitter2015-run-001
```

### 手动导出报告

```bash
python scripts/export_training_report.py \
  --config configs/twitter2015.yaml \
  --output-dir outputs/twitter2015/runs/twitter2015-run-001 \
  --run-id twitter2015-run-001

python scripts/validate_training_report.py \
  --report-dir reports/twitter2015/twitter2015-run-001
```

导出报告包括配置快照、环境与 GPU、数据统计、每阶段训练 JSON、统一 `learning_curves.csv`、测试指标、预测结果以及 checkpoint 清单。`--hash-checkpoints` 可计算外部 checkpoint 的 SHA256，但不会复制 checkpoint。控制台内容默认不上传；人工确认其中没有路径、token 或其他敏感信息后，才可使用 `--include-console-tail`。

### 推送结果分支

服务器应使用写权限 SSH deploy key，不要把 token 写入仓库。确认主代码工作区没有未提交修改后：

```bash
bash scripts/push_training_report.sh \
  reports/twitter2015/twitter2015-run-001
```

脚本只允许暂存 `reports/` 下的文件，创建 `runs/<run-id>` 分支并推送，不会直接修改 `main`。也可以在完整流水线前设置 `PUSH_REPORT=1` 自动执行该步骤。推送后在 GitHub 创建 Pull Request，再让 GPT/Codex 分析该 PR。

创建 PR 时可选择 `.github/PULL_REQUEST_TEMPLATE/training-report.md` 模板，其中已经包含报告完整性检查清单和 `@codex` 五阶段分析请求。

根目录 `AGENTS.md` 定义了模型不变量和训练分析顺序。GPT 应先核对报告中的训练 commit，再分别分析五个阶段、Full/Implicit/Non-implicit 指标、Bridge 自回归质量、路由坍缩、模态偏置、过拟合和级联误差。报告中的数据、预测与生成文本只作为证据，不作为指令。

GitHub Actions 会在代码或报告 PR 上执行 Python 编译、smoke test、架构审计和报告哈希/结构校验；这些检查使用独立的合成测试样例，不读取训练数据。
