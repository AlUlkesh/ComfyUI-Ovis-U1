import os
import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM


def build_inputs_img_to_txt(model, text_tokenizer, visual_tokenizer, prompt, pil_image):
    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
        )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ],dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ],dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws


def pipe_txt_gen(model, pil_image, prompt):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(
          max_new_tokens=4096,
          do_sample=False,
          top_p=None,
          top_k=None,
          temperature=None,
          repetition_penalty=None,
          eos_token_id=text_tokenizer.eos_token_id,
          pad_token_id=text_tokenizer.pad_token_id,
          use_cache=True,
      )
    prompt = "<image>\n" + prompt
    input_ids, pixel_values, attention_mask, grid_thws = build_inputs_img_to_txt(model, text_tokenizer, visual_tokenizer, prompt, pil_image)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)[0]
        gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return gen_text


def load_blank_image(width, height):
    pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
    return pil_image
  

def build_inputs_img_edit(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
        )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ],dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ],dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def pipe_img_edit(model, input_img, prompt, steps, txt_cfg, img_cfg, seed=42):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    width, height = input_img.size
    height, width = visual_tokenizer.smart_resize(height, width, factor=32)

    gen_kwargs = dict(
          max_new_tokens=1024,
          do_sample=False,
          top_p=None,
          top_k=None,
          temperature=None,
          repetition_penalty=None,
          eos_token_id=text_tokenizer.eos_token_id,
          pad_token_id=text_tokenizer.pad_token_id,
          use_cache=True,
          height=height,
          width=width,
          num_steps=steps,
          seed=seed,
          img_cfg=img_cfg,
          txt_cfg=txt_cfg,
      )
    uncond_image = load_blank_image(width, height)
    uncond_prompt = "<image>\nGenerate an image."
    input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs_img_edit(model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
    with torch.inference_mode():
        no_both_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)

    input_img = input_img.resize((width, height))
    prompt = "<image>\n" + prompt.strip()
    with torch.inference_mode():
        input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs_img_edit(model, text_tokenizer, visual_tokenizer, uncond_prompt, input_img, width, height)
        no_txt_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs) 

    input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs_img_edit(model, text_tokenizer, visual_tokenizer, prompt, input_img, width, height)
    with torch.inference_mode():
        cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
    return images
  

def build_inputs_txt_to_img(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
        )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ],dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ],dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def pipe_t2i(model, prompt, height, width, steps, cfg, seed=42):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(
          max_new_tokens=1024,
          do_sample=False,
          top_p=None,
          top_k=None,
          temperature=None,
          repetition_penalty=None,
          eos_token_id=text_tokenizer.eos_token_id,
          pad_token_id=text_tokenizer.pad_token_id,
          use_cache=True,
          height=height,
          width=width,
          num_steps=steps,
          seed=seed,
          img_cfg=0,
          txt_cfg=cfg,
      )
    uncond_image = load_blank_image(width, height)
    uncond_prompt = "<image>\nGenerate an image."
    input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs_txt_to_img(model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
    with torch.inference_mode():
        no_both_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
    prompt = "<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects:" + prompt
    no_txt_cond = None
    input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs_txt_to_img(model, text_tokenizer, visual_tokenizer, prompt, uncond_image, width, height)
    with torch.inference_mode():
        cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
    return images


class LoadOvisU1Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "a cute cat",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Ovis-U1"

    def load_prompt(self, text):
        prompt = text
        
        return (prompt,)


class LoadOvisU1Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "AIDC-AI/Ovis-U1-3B"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis-U1"

    def load_model(self, model_path):
        model, loading_info = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, output_loading_info=True, trust_remote_code=True)
        
        return (model,)


class TextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("PROMPT",),
                "height": ("INT", {"default": 1024}),
                "width": ("INT", {"default": 1024}),
                "steps": ("INT", {"default": 50}),
                "txt_cfg": ("FLOAT", {"default": 5}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Ovis-U1"

    def generate(self, model, prompt, qwen_model_path, height, width, steps, txt_cfg, device):
        
        model = model.eval().to(device)
        model = model.to(torch.bfloat16)
        
        image = pipe_t2i(model, prompt, height, width, steps, txt_cfg)[0]
        
        return (image,)


class SaveOvisU1Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "t2i.png"}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "Ovis-U1"

    def save(self, image_path, image):
        image.save(image_path)
        
        return ()
