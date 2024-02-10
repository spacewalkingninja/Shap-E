#!/usr/bin/env python

import gradio as gr
import spaces

from model import Model
from settings import CACHE_EXAMPLES, MAX_SEED
from utils import randomize_seed_fn


def create_demo(model: Model) -> gr.Blocks:
    examples = [
        "A chair that looks like an avocado",
        "An airplane that looks like a banana",
        "A spaceship",
        "A birthday cupcake",
        "A chair that looks like a tree",
        "A green boot",
        "A penguin",
        "Ube ice cream cone",
        "A bowl of vegetables",
    ]

    @spaces.GPU
    def process_example_fn(prompt: str) -> str:
        return model.run_text(prompt)

    @spaces.GPU
    def run(prompt: str, seed: int, guidance_scale: float, num_inference_steps: int) -> str:
        return model.run_text(prompt, seed, guidance_scale, num_inference_steps)

    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row(elem_id="prompt-container"):
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Model3D(label="Result", show_label=False)
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=20,
                    step=0.1,
                    value=15.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=2,
                    maximum=100,
                    step=1,
                    value=64,
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            fn=process_example_fn,
            cache_examples=CACHE_EXAMPLES,
        )

        gr.on(
            triggers=[prompt.submit, run_button.click],
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
            concurrency_limit=None,
        ).then(
            fn=run,
            inputs=[
                prompt,
                seed,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=result,
            api_name="text-to-3d",
            concurrency_id="gpu",
            concurrency_limit=1,
        )
    return demo
