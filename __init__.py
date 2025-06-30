from .nodes import LoadOvisU1Prompt, LoadOvisU1Model, TextToImage, SaveOvisU1Image

NODE_CLASS_MAPPINGS = {
    "LoadOvisU1Prompt": LoadOvisU1Prompt,
    "LoadOvisU1Model": LoadOvisU1Model,
    "TextToImage": TextToImage,
    "SaveOvisU1Image": SaveOvisU1Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOvisU1Prompt": "Load Ovis U1 Prompt",
    "LoadOvisU1Model": "Load Ovis U1 Model",
    "TextToImage": "Text To Image",
    "SaveOvisU1Image": "Save Ovis U1 Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
