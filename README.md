## CXR-LLaVA: Chest X-Ray Large Language and Vision Assistant

🥰 This project is based on the codebase of [LLaVA](https://llava-vl.github.io/) by Haotian Liu et al. Many thanks to them! As CXR-LLaVA is temporarily not released as a paper, please [cite their work](https://github.com/haotian-liu/LLaVA/tree/main#citation) if you are further developing on CXR-LLaVA.

🤗 **We have set up an [Online Demo](https://huggingface.co/spaces/TommyIX/CXR-LLaVA) on Huggingface**. Try it out!



#### Install Dependencies

1. Clone this repository and navigate to LLaVA folder
```shell
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
```shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. If you want to rerun the evaluation, please install two mandatory libraries using:

```shell
pip install pycocotools pycocoevalcap
```




#### Get the model weight

You can download the pretrained weight from [Huggingface: CXR-LLaVA-7b](https://huggingface.co/TommyIX/CXR-LLaVA-7b/) if you intend to do the evaluation.

As we are using [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) as the base model, you need to follow its instruction to get the model if you want to train CXR-LLaVA from scratch. We are using [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) or [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) as the vision tower, you can download each weight and change the directory in each training scripts to it. (Relative path is accepted)



#### Inference on CLI

CXR-LLaVA supports the inference pipeline on LLaVA. Inference on 16-bit can be run on a single 4090. You can refer to llava’s documentation for more details.

```Shell
python -m llava.serve.cli \
    --model-path TommyIX/CXR-LLaVA-7b \
    --image-file "path/to/your/image.jpg
```



#### Train

The pretraining alignment and finetuning are using same set of data. The tuning instructions json file can be downloaded using [this link](https://drive.google.com/file/d/1SfSzeL9eLJC3KqISz-pBz5tc4na5io1-/view?usp=sharing).

Please follow [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA/tree/2ca20de1ca76d7d121be5a53f8a46c6bef47a9cb#visual-instruction-tuning) part to download the required influencing natural data in `/data/images`, as saving the open-i data to `/data/openi-images`. So the data organization would be like:

```
images
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
├── vg
│   ├── VG_100K
│   └── VG_100K_2
├── p11  // These are the MIMIC-CXR data folders
├── p12
├── ...
├── p19
├── xxx1.png  // These are all Open-I raw images.
├── xxx2.png
└── ...
```

You can download OpenI data on [their official website]([openi.nlm.nih.gov/faq#collection](https://openi.nlm.nih.gov/faq#collection)). For the MIMIC-CXR data, please apply for usage on [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

You can use `scripts/pretrain7b.sh` and `scripts/finetune7b.sh` to conduct stage 1 and stage 2 training.



#### Evaluation

We are providing `eval_caption.py`,`mimic_caption.py` and `openi_caption.py` in the `eval` folder.  This covers all the experiments conducted in the report. You can specify the model directory and use them to reproduce the results.



#### Appendices

You are welcomed to open a issue or [send me an email](mailto:jinhong.wang@mbzuai.ac.ae) if you are encountering any problem in using MIMIC-CXR.
