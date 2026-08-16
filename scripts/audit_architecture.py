#!/usr/bin/env python3
"""Static contract audit; does not download pretrained models."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_method_args(tree: ast.Module, class_name: str, method_name: str) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return [arg.arg for arg in item.args.args]
    raise AssertionError(f"missing {class_name}.{method_name}")


def main() -> None:
    bridge = parse(ROOT / "src/tmis/models/bridge.py")
    args = class_method_args(bridge, "BridgeEncoderClassifier", "forward")
    assert args == ["self", "input_ids", "attention_mask"], (
        "Bridge classifier API must only consume bridge token IDs and attention mask; "
        f"got {args}"
    )

    encoders_text = (ROOT / "src/tmis/models/encoders.py").read_text(encoding="utf-8")
    assert "T5ForConditionalGeneration" in encoders_text
    assert "CLIPVisionModel" in encoders_text
    assert "AutoModel" not in encoders_text

    bridge_text = (ROOT / "src/tmis/models/bridge.py").read_text(encoding="utf-8")
    assert "encoder_outputs=encoder_outputs" in bridge_text
    assert "t5_model" in bridge_text
    assert "t5_model.generate(" in bridge_text
    assert "use_cache=True" in bridge_text
    assert "memory_type_embeddings" in bridge_text
    assert "pretrained_token_embeddings" in bridge_text
    assert "nn.TransformerDecoder" not in bridge_text, (
        "reasoning bridge generation must use the pretrained T5 decoder, "
        "not a separate randomly initialized TransformerDecoder"
    )

    config_texts = {
        name: (ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8")
        for name in ("twitter2015", "twitter2017")
    }
    for name, config_text in config_texts.items():
        assert "text_backbone: google-t5/t5-large" in config_text, name
        assert "vision_backbone: openai/clip-vit-large-patch14" in config_text, name
        assert "text_backbone_revision:" in config_text, name
        assert "vision_backbone_revision:" in config_text, name

    constants_text = (ROOT / "src/tmis/constants.py").read_text(encoding="utf-8")
    assert 'BRIDGE_EOS_TOKEN = "</s>"' in constants_text

    collator_text = (ROOT / "src/tmis/data/collator.py").read_text(encoding="utf-8")
    assert "target_input_ids" in collator_text
    assert '"token_type_ids"' not in collator_text

    trainer_text = (ROOT / "src/tmis/training/trainer.py").read_text(encoding="utf-8")
    for stage in [
        "stage1_aux",
        "stage2_reasoning_warmup",
        "stage3_bridge",
        "stage4_classifier",
        "stage5_joint",
    ]:
        assert stage in trainer_text, f"missing training stage: {stage}"
    assert "generate_bridge" in trainer_text
    assert "generated_bridge_ratio" in trainer_text
    assert "routing_gold_mix" in trainer_text
    assert "def evaluate_bridge_generation" in trainer_text
    assert 'self.out_dir / "best_bridge.pt"' in trainer_text
    assert "routing_gold_mix=0.0" in trainer_text
    assert "best_meta = load_checkpoint(" in trainer_text
    assert "DistributedDataParallel" in trainer_text
    assert "DistributedSampler" in trainer_text
    assert "torch.amp.GradScaler" in trainer_text
    assert '"stage5_generated_only.pt"' in trainer_text
    assert '"latest.pt"' in trainer_text
    assert "visual_negative_bank" in trainer_text
    assert "text_evidence_focal_gamma" in trainer_text

    for name, config_text in config_texts.items():
        assert "selection_metric: rouge_l_f1_full" in config_text, name
        assert "generated_bridge_ratio_end: 1.0" in config_text, name

    model_text = (ROOT / "src/tmis/models/model.py").read_text(encoding="utf-8")
    assert "compute_visual_evidence_target: bool = False" in model_text
    assert "if routing_gold_mix > 0 else None" in model_text
    assert "self.text_encoder.backbone" in model_text
    assert "def forward(" in model_text
    assert "visual_presence_logits" in model_text
    assert model_text.index("h_rel = self.relation") < model_text.index("paths = self.reasoner"), (
        "Cross-modal Head must run before the three reasoning paths"
    )

    loss_text = (ROOT / "src/tmis/training/losses.py").read_text(encoding="utf-8")
    for name in [
        "text_evidence_loss",
        "visual_evidence_contrastive_loss",
        "reasoning_tag_loss",
        "bridge_generation_loss",
        "sentiment_loss",
    ]:
        assert f"def {name}" in loss_text

    checkpoint_text = (ROOT / "src/tmis/utils/checkpoint.py").read_text(encoding="utf-8")
    assert "os.replace" in checkpoint_text
    assert "weights_only=True" in checkpoint_text

    dataset_text = (ROOT / "src/tmis/data/dataset.py").read_text(encoding="utf-8")
    assert "candidate.relative_to(self.image_dir)" in dataset_text

    print("ARCHITECTURE_AUDIT_OK")


if __name__ == "__main__":
    main()
