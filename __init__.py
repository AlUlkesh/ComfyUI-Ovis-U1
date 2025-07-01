from .nodes import LoadOvisU1Prompt, LoadOvisU1Image, LoadOvisU1Model, TextToImage, ImageEdit, ImageToText, SaveOvisU1Image

NODE_CLASS_MAPPINGS = {
    "LoadOvisU1Prompt": LoadOvisU1Prompt,
    "LoadOvisU1Image": LoadOvisU1Image,
    "LoadOvisU1Model": LoadOvisU1Model,
    "TextToImage": TextToImage,
    "ImageEdit": ImageEdit,
    "ImageToText": ImageToText,
    "SaveOvisU1Image": SaveOvisU1Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOvisU1Prompt": "Load Ovis U1 Prompt",
    "LoadOvisU1Image": "Load Ovis U1 Image",
    "LoadOvisU1Model": "Load Ovis U1 Model",
    "TextToImage": "Text To Image",
    "ImageEdit": "Image Edit",
    "ImageToText": "Image To Text",
    "SaveOvisU1Image": "Save Ovis U1 Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
