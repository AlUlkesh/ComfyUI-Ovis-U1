from .nodes import LoadOvisU1Prompt, LoadOvisU1Model, OvisU1TextToImage, OvisU1ImageEdit, OvisU1ImageToText

NODE_CLASS_MAPPINGS = {
    "LoadOvisU1Prompt": LoadOvisU1Prompt,
    "LoadOvisU1Model": LoadOvisU1Model,
    "OvisU1TextToImage": OvisU1TextToImage,
    "OvisU1ImageEdit": OvisU1ImageEdit,
    "OvisU1ImageToText": OvisU1ImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOvisU1Prompt": "Load Ovis U1 Prompt",
    "LoadOvisU1Model": "Load Ovis U1 Model",
    "OvisU1TextToImage": "Ovis U1 Text To Image",
    "OvisU1ImageEdit": "Ovi sU1 Image Edit",
    "OvisU1ImageToText": "Ovis U1 Image To Text",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
