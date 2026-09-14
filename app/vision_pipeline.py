"""Vision Pipeline combining Pillow/OpenCV Preprocessor, PyTorch CLIP, and Keras MobileNetV2."""

import os
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np

# Configure Keras backend to PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import torch
from transformers import CLIPModel, CLIPProcessor
import keras
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

from app.config import settings
try:
    from app.image_preprocessor import ImagePreprocessor
except ImportError:
    from image_preprocessor import ImagePreprocessor



class VisionPipeline:
    """End-to-end vision pipeline combining preprocessing and dual deep vision models:

    1. Pillow/OpenCV ImagePreprocessor (CLAHE + 224x224 aspect-preserving letterboxing)
    2. PyTorch Hugging Face Transformers CLIP model (Visual embeddings for vector search)
    3. Keras MobileNetV2 (1280-d deep visual feature representations)
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. Initialize Pillow/OpenCV preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=settings.TARGET_IMAGE_SIZE,
            clip_limit=settings.CLAHE_CLIP_LIMIT,
            tile_grid_size=settings.CLAHE_TILE_GRID_SIZE,
            pad_color=(0, 0, 0),
        )

        # 2. Initialize PyTorch CLIP model
        print(f"Loading CLIP model ({settings.CLIP_MODEL_NAME}) on {self.device}...")
        self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.clip_model.eval()

        # 3. Initialize Keras MobileNetV2
        print("Loading Keras MobileNetV2 (weights='imagenet', pooling='avg')...")
        self.mobilenet_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )

        print("VisionPipeline successfully initialized!")

    def preprocess_image(self, image_input: Union[Image.Image, np.ndarray, str, bytes]) -> Image.Image:
        """Preprocesses image through CLAHE enhancement and aspect-ratio padding to 224x224."""
        return self.preprocessor.preprocess(image_input)

    def extract_clip_visual_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract L2-normalized 512-dimensional visual embedding vector using PyTorch CLIP.

        Args:
            image: Preprocessed PIL Image (224x224).

        Returns:
            np.ndarray: 1D normalized float32 embedding vector of length 512.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            output = self.clip_model.get_image_features(pixel_values=pixel_values)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                image_features = output.pooler_output
            else:
                image_features = output

            # L2 normalize using PyTorch
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        embedding = image_features.detach().cpu().numpy().squeeze(0).astype(np.float32)
        return embedding

    def extract_clip_text_embedding(self, text: str) -> np.ndarray:
        """Extract L2-normalized 512-dimensional text embedding vector using PyTorch CLIP.

        Useful for text queries and catalog item initialization.
        """
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                text_features = output.pooler_output
            else:
                text_features = output

            # L2 normalize using PyTorch
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        embedding = text_features.detach().cpu().numpy().squeeze(0).astype(np.float32)
        return embedding

    def extract_mobilenet_features(self, image: Image.Image) -> np.ndarray:
        """Extract 1280-dimensional feature vector using Keras MobileNetV2.

        Args:
            image: Preprocessed PIL Image (224x224).

        Returns:
            np.ndarray: 1D float32 feature array of length 1280.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # Standard MobileNetV2 input preprocessing (maps to [-1, 1])
        preprocessed = mobilenet_preprocess(img_batch)

        with torch.no_grad():
            features = self.mobilenet_model(preprocessed, training=False)
            if hasattr(features, "detach"):
                features_np = features.detach().cpu().numpy().squeeze(0).astype(np.float32)
            else:
                features_np = np.asarray(features).squeeze(0).astype(np.float32)

        # L2 normalize for stable distance metrics
        norm = np.linalg.norm(features_np)
        if norm > 1e-6:
            features_np = features_np / norm

        return features_np

    def process_and_extract(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes],
        extract_mobilenet: bool = True,
    ) -> Tuple[Image.Image, np.ndarray, Optional[np.ndarray]]:
        """Complete visual processing pipeline:

        1. Passes image through CLAHE + 224x224 aspect-preserving pad preprocessor.
        2. Computes PyTorch CLIP 512-d visual embedding.
        3. Optionally computes Keras MobileNetV2 1280-d features.

        Returns:
            Tuple[Image.Image, np.ndarray, Optional[np.ndarray]]:
                (preprocessed_image, clip_embedding, mobilenet_features)
        """
        preprocessed_img = self.preprocess_image(image_input)
        clip_emb = self.extract_clip_visual_embedding(preprocessed_img)

        mobilenet_feat = None
        if extract_mobilenet:
            mobilenet_feat = self.extract_mobilenet_features(preprocessed_img)

        return preprocessed_img, clip_emb, mobilenet_feat
