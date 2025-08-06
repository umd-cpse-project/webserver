from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

import clip
import torch
from aiosqlite import Row
from PIL import Image

# Load the model (once)
model, preprocess = clip.load("ViT-L/14", device="cpu")

__all__ = (
    'Category', 
    'CategorizationResponse', 
    'CategoryManifest',
)


@dataclass
class Category:
    id: int
    position: tuple[int, int]  # (x, y) coordinates
    name: str
    context: str = ''  # used to provide additional context
    size: tuple[int, int] = (1, 1)  # (width, height)
    hue: int = 0

    @property
    def x(self) -> int:
        return self.position[0]
    
    @property
    def y(self) -> int:
        return self.position[1]

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]
    
    @classmethod
    def dummy(cls, name: str, *, context: str | None = None) -> Self:
        return cls(id=-1, position=(0, 0), name=name, context=context or name)

    @classmethod
    def from_record(cls, record: Row) -> Self:
        return cls(
            id=record['id'],
            position=(record['x'], record['y']),
            name=record['name'],
            context=record['context'],
            size=(record['width'], record['height']),
            hue=record['hue'],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'name': self.name,
            'context': self.context,
            'width': self.width,
            'height': self.height,
            'hue': self.hue,
        }
    
    @property
    def graphic_path(self) -> str:
        return f'static/graphics/{self.id}.png'
    
    def set_graphic(self, graphic: bytes) -> str:
        with open(self.graphic_path, 'wb') as f:
            f.write(graphic)
        return self.graphic_path
    
    def remove_graphic(self) -> None:
        try:
            os.remove(self.graphic_path)
        except FileNotFoundError:
            pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<Category id={self.id} position={self.position} name={self.name!r}>'


class CategorizationResponse(NamedTuple):
    category: Category
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'category': self.category.to_dict(),
            'confidence': self.confidence,
        }


class CategoryManifest(NamedTuple):
    categories: list[Category]
    inputs: torch.Tensor

    @classmethod
    def from_categories(cls, categories: list[Category]) -> Self:
        inputs = clip.tokenize([c.context or c.name for c in categories]).to("cpu")
        return cls(categories, inputs)

    def categorize(self, image: Image.Image) -> CategorizationResponse:
        # Load and preprocess the image
        img = preprocess(image).unsqueeze(0).to("cpu")

        # Perform inference
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(self.inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            conf, idx = similarity.max(0)

        return CategorizationResponse(self.categories[idx.item()], conf.item())


def start_debug_categorization(manifest: CategoryManifest) -> None:
    import cv2

    capture = cv2.VideoCapture(0)
    last_prediction_response: CategorizationResponse | None = None
    downscale_width: int = 224

    def on_mouse_click(event, _x, _y, _flags, image):
        nonlocal last_prediction_response
        
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imwrite("frame.png", image)

            image = Image.open('frame.png').convert("RGB").resize(
                (downscale_width, round(downscale_width * aspect_ratio)))
            response = last_prediction_response = manifest.categorize(image)
            print(response.category, response.confidence)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        aspect_ratio = frame.shape[1] / frame.shape[0]
        if last_prediction_response:
            category = last_prediction_response.category
            cv2.putText(
                frame, 
                f'{category.name} ({last_prediction_response.confidence:.2f})',
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5
            )
            
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', on_mouse_click, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()


if __name__ == '__main__':
    categories = [
        Category.dummy(s) for s in ['water bottle', 'mouse', 'phone', 'resistor']
    ]
    start_debug_categorization(CategoryManifest.from_categories(categories))
