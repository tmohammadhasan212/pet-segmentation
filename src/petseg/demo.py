import os
import tempfile
from pathlib import Path

from .inference import Predictor, export


def create_demo(checkpoint, device="auto"):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    predictor = Predictor(checkpoint, device)

    def remove(image, threshold):
        if image is None:
            raise gr.Error("Upload a pet photograph first.")
        try:
            result = predictor.predict(image, threshold)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        directory = Path(tempfile.mkdtemp(prefix="petseg-demo-"))
        export(result, directory)
        files = [
            str(directory / name)
            for name in ("cutout.png", "mask.png", "probability.png", "probability.npy")
        ]
        status = f"Model: {result['version']} · Threshold: {threshold:.2f} · Inference: {result['inference_ms']:.1f} ms · Device: {predictor.device}"
        return (
            result["original"],
            result["heatmap"],
            result["binary"],
            result["cutout"],
            status,
            files,
        )

    with gr.Blocks(title="Pet foreground studio", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# Pet foreground studio\nUpload a cat or dog photograph and compare the predicted foreground. Images without a pet can produce false positives. Fine fur, tails, small pets and multiple animals can be unreliable."
        )
        with gr.Row():
            upload = gr.Image(
                type="filepath", label="Pet photograph", sources=["upload"]
            )
            with gr.Column():
                threshold = gr.Slider(
                    0,
                    1,
                    value=predictor.checkpoint.get("threshold", 0.5),
                    step=0.01,
                    label="Foreground threshold",
                )
                run = gr.Button("Remove background", variant="primary")
                status = gr.Markdown(f"Model: {predictor.version}")
        with gr.Row():
            original = gr.Image(label="Original", interactive=False)
            probability = gr.Image(
                label="Pet probability · blue 0 → yellow 1", interactive=False
            )
        with gr.Row():
            binary = gr.Image(label="Binary mask", interactive=False)
            cutout = gr.Image(
                label="Transparent PNG", image_mode="RGBA", interactive=False
            )
        files = gr.File(
            label="Download cutout, masks and raw probabilities", file_count="multiple"
        )
        run.click(
            remove,
            [upload, threshold],
            [original, probability, binary, cutout, status, files],
            concurrency_limit=1,
        )
        gr.Markdown(
            "The model is trained on Oxford-IIIT Pet trimaps. Its ambiguous annotation border is ignored during training and scoring. A binary alpha cutout does not recover individual translucent hairs."
        )
    return demo
