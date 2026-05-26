import torch


def export_onnx(model, sample_input, output_path):
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
