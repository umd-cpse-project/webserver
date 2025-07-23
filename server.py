from __future__ import annotations

import asyncio
import base64
import logging
import msgpack
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated, Any, Final

import pydantic
import uvicorn
from dotenv import load_dotenv
from PIL import Image
from fastapi import Depends, FastAPI, File, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_mqtt import FastMQTT, MQTTClient, MQTTConfig

from database import DbConnection

MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10 MB
MAX_GRAPHIC_SIZE: Final[int] = 2 * 1024 * 1024  # 2 MB
MIN_IMAGE_DIMENSION: Final[int] = 64  # 64 pixels

load_dotenv()
TOKEN = os.getenv('AUTHORIZATION_TOKEN')
WEB_ACCESS_PASSWORD = os.getenv('WEB_ACCESS_PASSWORD')

mqtt_config = MQTTConfig(
    host=os.getenv('MQTT_HOST'),
    port=int(os.getenv('MQTT_PORT')),
    username=os.getenv('MQTT_USERNAME'),
    password=os.getenv('MQTT_PASSWORD'),
    ssl=True,
)
mqtt = FastMQTT(config=mqtt_config, client_id='webserver')
db = DbConnection()
log = logging.getLogger('uvicorn.error')


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await mqtt.mqtt_startup()
    yield
    await mqtt.mqtt_shutdown()


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@mqtt.on_connect()
def connect(client: MQTTClient, _flags: int, _rc: int, _properties: Any):
    client.subscribe("/webcam")
    log.info('mqtt: client connected')


@mqtt.on_message()
async def message(client: MQTTClient, topic: str, payload: bytes, _qos, _properties):
    if topic.startswith('/webcam'):
        try:
            with open(f'static/last_webcam_update.jpg', 'wb') as fp:
                fp.write(payload)
                
            image = Image.open(BytesIO(payload)).convert('RGB')
            if manifest := await db.fetch_manifest():
                response = await asyncio.to_thread(manifest.categorize, image)
                log.info(
                    f'Webcam image processed: {response.category} '
                    f'with confidence {response.confidence:.2f}'
                )            
                payload = msgpack.dumps(response.to_dict())
                client.publish('/categorization', payload)
                
        except Exception as e:
            log.error(f"Error processing image from topic '{topic}': {e}")
    return 0


@mqtt.on_disconnect()
def disconnect(_client, _packet, exc=None):
    log.info('mqtt: client disconnected', _client, _packet, exc)


@mqtt.on_subscribe()
def subscribe(client, mid, qos, properties):
    log.debug('mqtt: subscription create:', client, mid, qos, properties)


def verify_token(authorization: str = Header()):
    if authorization != WEB_ACCESS_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid or missing web access password")


AuthorizationRequired = lambda: Depends(verify_token)


@app.get('/')
async def root():
    return {'message': 'Hello, world!'}


@app.post('/categorize')
async def categorize(
    _: Annotated[Any, AuthorizationRequired()],
    file: bytes = File(min_length=0, max_length=MAX_FILE_SIZE),
):
    """Takes a multipart image upload and returns a categorization response."""
    image = Image.open(BytesIO(file)).convert('RGB')
    if image.width < MIN_IMAGE_DIMENSION or image.height < MIN_IMAGE_DIMENSION:
        raise RequestValidationError(
            f"Image dimensions must be at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
        )

    manifest = await db.fetch_manifest()
    if manifest is None:
        raise RequestValidationError("No categories available for categorization.")
    
    response = await asyncio.to_thread(manifest.categorize, image)
    return {
        'confidence': response.confidence,
        'category': response.category,
    }


@app.get('/categories', response_model=list[dict])
async def get_categories(_: Annotated[Any, AuthorizationRequired()],):
    """Fetch all categories from the database."""
    categories = await db.fetch_categories()
    return [category.to_dict() for category in categories]


@app.get('/categories/{category_id}', response_model=dict)
async def get_category(
    _: Annotated[Any, AuthorizationRequired()],
    category_id: int,
):
    """Fetch a single category by its ID."""
    category = await db.fetch_category(category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return category.to_dict()


class CreateCategoryPayload(pydantic.BaseModel):
    name: str
    context: str
    x: int
    y: int
    graphic: str | None = None
    width: int = 1
    height: int = 1
    hue: int = 0


class EditCategoryPayload(pydantic.BaseModel):
    name: str | None = None
    context: str | None = None
    x: int | None = None
    y: int | None = None
    graphic: str | None = None
    width: int | None = None
    height: int | None = None
    hue: int | None = None


@app.post('/categories', status_code=201)
async def create_category(
    _: Annotated[Any, AuthorizationRequired()],
    payload: CreateCategoryPayload,
):
    """Create a new category with the provided details."""
    graphic = payload.graphic and base64.b64decode(payload.graphic)
    if graphic and len(graphic) > MAX_GRAPHIC_SIZE:
        raise RequestValidationError(f"Graphic size must not exceed {MAX_GRAPHIC_SIZE} bytes.")

    category = await db.create_category(
        name=payload.name,
        context=payload.context,
        position=(payload.x, payload.y),
        size=(payload.width, payload.height),
        hue=payload.hue,
        graphic=graphic,
    )
    return category.to_dict()


@app.patch('/categories/{category_id}', status_code=200)
async def edit_category(
    _: Annotated[Any, AuthorizationRequired()],
    category_id: int,
    payload: EditCategoryPayload,
):
    """Edit an existing category with the provided details.
    
    Note: both `x` and `y` must be provided together, as well as `width` and `height`.
    """
    category = await db.edit_category(
        category_id,
        name=payload.name,
        context=payload.context,
        position=(payload.x, payload.y) if payload.x is not None and payload.y is not None else None,
        size=(payload.width, payload.height) if payload.width is not None and payload.height is not None else None,
        hue=payload.hue,
    )
    
    if 'graphic' in payload.model_fields_set:
        if payload.graphic is None:
            category.remove_graphic()
        else:
            graphic = payload.graphic and base64.b64decode(payload.graphic)
            if len(graphic) > MAX_GRAPHIC_SIZE:
                raise RequestValidationError(f"Graphic size must not exceed {MAX_GRAPHIC_SIZE} bytes.")
            category.set_graphic(graphic)
    
    return category.to_dict()


@app.delete('/categories/{category_id}', status_code=204)
async def delete_category(
    _: Annotated[Any, AuthorizationRequired()],
    category_id: int,
):
    """Delete a category by its ID."""
    await db.delete_category(category_id)
    return {"message": "Category deleted successfully."}


@app.patch('/categories', status_code=200)
async def set_categories(
    _: Annotated[Any, AuthorizationRequired()],
    payload: list[EditCategoryPayload],
):
    """Clears all categories and replaces them with the provided list of categories."""
    return [c.to_dict() for c in await db.set_categories(payload)]


@app.delete('/categories', status_code=204)
async def delete_all_categories(
    _: Annotated[Any, AuthorizationRequired()],
):
    """Delete all categories from the database."""
    await db.clear_categories()
    return {"message": "All categories deleted successfully."}


if __name__ == '__main__':
    uvicorn.run(app, log_level="trace")
