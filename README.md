# ComfyUI-Ovis-U1

ComfyUI-Ovis-U1 is now available in ComfyUI, [Ovis-U1](https://github.com/AIDC-AI/Ovis-U1) is a 3-billion-parameter unified model that seamlessly integrates multimodal understanding, text-to-image generation, and image editing within a single powerful framework.



## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-Ovis-U1.git
```

3. Install dependencies:
```
cd ComfyUI-Ovis-U1

# Create conda environment
conda create -n ovis-u1 python=3.10 -y
conda activate ovis-u1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```


## Model

### Download Pretrained Models

**Ovis-U1-3B**, a multimodal model, model weights can be accessed in [huggingface](https://huggingface.co/AIDC-AI/Ovis-U1-3B).


## 🏆 Highlights

*   **Unified Capabilities**: A single model excels at three core tasks: understanding complex scenes, generating images from text, and performing precise edits based on instructions.
*   **Advanced Architecture**: Ovis-U1 features a powerful diffusion-based visual decoder (MMDiT) and a bidirectional token refiner, enabling high-fidelity image synthesis and enhanced interaction between text and vision.
*   **Synergistic Unified Training**: Unlike models trained on single tasks, Ovis-U1 is trained on a diverse mix of understanding, generation, and editing data simultaneously. Our findings show that this approach achieves improved generalization, seamlessly handling real-world multimodal challenges with high accuracy.
*   **State-of-the-Art Performance**: Ovis-U1 achieves leading scores on multiple academic benchmarks, surpassing strong contemporary models in multimodal understanding (69.6 on OpenCompass), generation (83.72 on DPG-Bench), and editing (4.00 on ImgEdit-Bench).

