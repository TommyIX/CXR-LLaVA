from eval_caption import OpenIEvalCap
import json
import torch
import torch.utils.data
from torch.utils.data import Dataloader
import numpy as np
import tqdm
from PIL import Image
import os
from ..llava.model.builder import load_pretrained_model
from ..llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from ..llava.conversation import conv_templates, SeparatorStyle 
from ..llava.utils import disable_torch_init
from ..llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ..llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN



def sample_dataset(dataset, max_sample=1000, seed=0):
    if max_sample == -1:
        return dataset
    
    if len(dataset) > max_sample:
        np.random.seed(seed)
        rand_indices = np.random.choice(
            len(dataset), max_sample, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, rand_indices)
        return dataset
    
def llava_generate(image_path, question, model, tokenizer, image_processor):
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
                
    prompt = question
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device))
                
                
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images = image_tensor
        )
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    return outputs

def calculate_metrics(model,
                      tokenizer,
                      image_processor,
                      dataset,
                      question="Describe the given chest x-ray image in detail.",
                      answer_path="/lustre/home/yongxin.wang/workspace/LLaVA/playground/data/eval/openi"):
    
    pred = []
    dataloader = Dataloader(dataset, batch_size=1, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = llava_generate(batch['image_path'], question, model, tokenizer, image_processor)
        for image_path, gt, output in zip(batch["image_path"], batch["caption"], outputs):
            answer_dict = {
                'image_id': image_path,
                'caption': gt,
                'answer': output
            }
            pred.append(answer_dict)
    answer_dir = os.path.join(answer_path, "llava-med")
    os.makedirs(answer_dir, exist_ok=True)
    answer_path_dataset = os.path.join(answer_dir, f"{'openI'}.json")
    with open(answer_path_dataset, 'w') as f:
        f.write(json.dumps(pred, indent=4))
        
    gts = {}
    res = {}
        
    with open(answer_path_dataset, 'r') as f:
        answer = json.load(f)
    
    img2gts = {ann['image_id']: [] for ann in answer }
    for ann in answer:
        img2gts[ann['image_id']] += [ann['caption']]
        
    img2res = {ann['image_id']: [] for ann in answer }
    for ann in answer:
        img2res[ann['image_id']] += [ann['answer']]
        
    img_id = range(1000)
    for id in img_id:
        gts[id] = img2gts[id]
        res[id] = img2res[id]
        
    eval = OpenIEvalCap(img_id, gts, res)
    eval.evaluate()
    
    return eval.eval



if __name__ == '__main__':
    eval_dataset = sample_dataset("OpenI")
    model_path = '../checkpoints/llava1.5-MLM7b-ft/ '
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
    llava_generate(model, tokenizer, image_processor, eval_dataset)
    
    