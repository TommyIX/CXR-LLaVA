## CXR-LLaVA: Chest X-Ray Large Language and Vision Assistant

🥰 This project is based on the codebase of [LLaVA](https://llava-vl.github.io/) by Haotian Liu et al. Many thanks to them! As CXR-LLaVA is temporarily not released as a paper, please [cite their work](https://github.com/haotian-liu/LLaVA/tree/main#citation) if you are further developing on CXR-LLaVA.

🤗 **We have set up an [Online Demo]([CXR LLaVA - a Hugging Face Space by TommyIX](https://huggingface.co/spaces/TommyIX/CXR-LLaVA)) on Huggingface**. Try it out!



#### Install Dependencies

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/TommyIX/CXR-LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install open-clip-torch
```

3. If you are going to train the model, you need to run:
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```



#### Get the model weight

You can download the pretrained weight from [Huggingface: CXR-LLaVA-7b](https://huggingface.co/TommyIX/CXR-LLaVA-7b/). As we are using [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) as the base model, 



#### Inference on CLI

CXR-LLaVA supports the inference pipeline on LLaVA. Inference on 16-bit can be run on a single 4090. You can refer to llava’s documentation for more details.

```Shell
python -m llava.serve.cli \
    --model-path TommyIX/CXR-LLaVA-7b \
    --image-file "path/to/your/image.jpg
```



#### Train

TBF



#### Evaluation

TBF
