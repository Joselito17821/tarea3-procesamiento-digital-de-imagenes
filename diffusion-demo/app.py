import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random

# ==========================================
# CONFIGURACIÓN
# ==========================================
HF_TOKEN = "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"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

PROMPTS = {
    "🎨 Arte": [
        "a surrealist painting of a melting clock in a desert, Salvador Dali style",
        "Van Gogh style painting of a starry night over a futuristic city",
        "abstract expressionist painting, vibrant colors, dynamic brushstrokes",
    ],
    "🔬 Ciencia": [
        "glowing DNA double helix floating in a dark laboratory, highly detailed",
        "neural network visualization, glowing nodes and connections, digital art",
        "microscopic view of a neuron firing, bioluminescent, ultra detailed",
    ],
    "🏙️ Futurista": [
        "cyberpunk city at night, neon lights, rain reflections, blade runner style",
        "futuristic space station orbiting Earth, cinematic lighting, 8k",
        "robot hand touching human hand, digital art, dramatic lighting",
    ],
    "🌿 Naturaleza": [
        "enchanted forest with magical glowing mushrooms, fantasy art, misty",
        "underwater coral reef with bioluminescent creatures, ultra HD",
        "aurora borealis over snowy mountains, photorealistic, stunning",
    ],
}

# ==========================================
# CARGAR MODELO
# ==========================================
print("⏳ Cargando modelo... (primera vez tarda varios minutos)")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    scheduler=DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler"),
)

pipe = pipe.to("cuda")
pipe.enable_attention_slicing(1)
pipe.vae.enable_slicing()

print("✅ Modelo cargado correctamente")

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================
def generate_image(prompt, category, steps, guidance, seed, use_random_prompt):

    if use_random_prompt and category in PROMPTS:
        prompt = random.choice(PROMPTS[category])

    if not prompt.strip():
        return None, "⚠️ Escribe un prompt o activa el prompt aleatorio"

    generator = torch.Generator("cuda")
    if seed == -1:
        seed = random.randint(0, 999999)
    generator.manual_seed(int(seed))

    print(f"🎨 Generando: '{prompt}' | Steps: {steps} | CFG: {guidance} | Seed: {seed}")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
            height=512,
            width=512,
        )

    image = result.images[0]
    info = f"✅ Generado | Prompt: '{prompt}' | Steps: {steps} | CFG: {guidance} | Seed: {seed}"
    return image, info

# ==========================================
# INTERFAZ GRADIO
# ==========================================
with gr.Blocks(css="""
    .container { max-width: 1100px; margin: auto; }
    .title { text-align: center; margin-bottom: 20px; }
    footer { display: none !important; }
""") as demo:

    gr.Markdown(
        """
        # 🌌 Diffusion Models — Demo en Vivo
        ### Generación de imágenes con Stable Diffusion v1.5
        *Tarea 3 — Procesamiento Digital de Imágenes*
        """,
        elem_classes="title"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Parámetros")

            prompt_input = gr.Textbox(
                label="📝 Prompt",
                placeholder="Describe la imagen que quieres generar...",
                lines=3
            )

            category = gr.Dropdown(
                choices=list(PROMPTS.keys()),
                value="🏙️ Futurista",
                label="🎭 Categoría"
            )

            use_random = gr.Checkbox(
                label="🎲 Usar prompt aleatorio de la categoría",
                value=False
            )

            steps = gr.Slider(
                minimum=10, maximum=50, value=20, step=5,
                label="🔄 Pasos de inferencia (más = mejor calidad, más lento)"
            )

            guidance = gr.Slider(
                minimum=1, maximum=20, value=7.5, step=0.5,
                label="🎯 Guidance Scale (qué tan fiel al prompt)"
            )

            seed = gr.Number(
                value=-1,
                label="🌱 Seed (-1 = aleatorio)"
            )

            btn = gr.Button("🚀 Generar imagen", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Resultado")
            output_image = gr.Image(label="Imagen generada", height=500)
            output_info = gr.Textbox(label="ℹ️ Info de generación", interactive=False)

    gr.Markdown("### 💡 Ejemplos rápidos")
    gr.Examples(
        examples=[
            ["cyberpunk city at night, neon lights, rain, blade runner style", "🏙️ Futurista", 20, 7.5, 42, False],
            ["glowing DNA double helix in a laboratory, highly detailed", "🔬 Ciencia", 20, 8.0, 123, False],
            ["Van Gogh style starry night over a futuristic city", "🎨 Arte", 20, 7.0, 777, False],
            ["enchanted forest with magical glowing mushrooms, misty", "🌿 Naturaleza", 20, 7.5, 999, False],
        ],
        inputs=[prompt_input, category, steps, guidance, seed, use_random],
        label="Haz clic en cualquier ejemplo para cargarlo"
    )

    btn.click(
        fn=generate_image,
        inputs=[prompt_input, category, steps, guidance, seed, use_random],
        outputs=[output_image, output_info]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="purple",
            neutral_hue="slate",
        )
    )