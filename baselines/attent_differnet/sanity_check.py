from __future__ import annotations

import torch
from torchvision.models import AlexNet_Weights, alexnet

from .attention import CBAM, SqueezeExcitation
from .model import AlexNetFeaturesWithAttention, AttentDifferNet, DifferNetConfig


def main() -> int:
    x = torch.randn(2, 3, 448, 448)
    reference = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features.eval()
    baseline = AlexNetFeaturesWithAttention(attention="none", pretrained=True).eval()
    with torch.no_grad():
        y_reference = reference(x)
        y_baseline = baseline(x)

    print("none feature shape:", tuple(y_baseline.shape))
    print("max abs diff vs torchvision alexnet.features:", float((y_reference - y_baseline).abs().max()))

    for attention in ["none", "se", "cbam"]:
        model = AttentDifferNet(DifferNetConfig(attention=attention, pretrained=False))
        feature_trainable = sum(p.numel() for p in model.feature_extractor.parameters() if p.requires_grad)
        nf_trainable = sum(p.numel() for p in model.nf.parameters() if p.requires_grad)
        opt_count = sum(p.numel() for p in model.optim_parameters())
        print(
            attention,
            "feature_trainable=",
            feature_trainable,
            "nf_trainable=",
            nf_trainable,
            "optimizer_params=",
            opt_count,
        )

    for attention, module_type in [("se", SqueezeExcitation), ("cbam", CBAM)]:
        model = AlexNetFeaturesWithAttention(attention=attention, pretrained=False).eval()
        print(f"{attention} attention positions:")
        hooks = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                print(" ", name, type(module).__name__, tuple(inputs[0].shape), "->", tuple(output.shape))

            return hook

        for index, module in enumerate(model.features):
            if isinstance(module, module_type):
                hooks.append(module.register_forward_hook(make_hook(f"features.{index}")))
        with torch.no_grad():
            model(torch.randn(1, 3, 448, 448))
        for hook in hooks:
            hook.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
