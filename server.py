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

BOARD_WIDTH: Final[int] = 32
BOARD_HEIGHT: Final[int] = 24

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
            payload = {'error': str(e), 'context': 'process webcam image'}
            client.publish('/error/categorization', msgpack.dumps(payload))

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


@app.get('/login')
async def login(
    _: Annotated[Any, AuthorizationRequired()],
):
    """Endpoint to verify the web access password, and returns MQTT broker connection (over socket) details."""
    return {
        'mqtt_connect_url': os.getenv('MQTT_WEB_BROKER'),
        'mqtt_username': mqtt.config.username,
        'mqtt_password': mqtt.config.password,
    }


@app.post('/categorize')
async def categorize(
    _: Annotated[Any, AuthorizationRequired()],
    image: bytes = File(min_length=0, max_length=MAX_FILE_SIZE),
):
    """Takes a multipart image upload and returns a categorization response."""
    image = Image.open(BytesIO(image)).convert('RGB')
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


async def validate_category_payload(
    payload: CreateCategoryPayload | EditCategoryPayload,
    *,
    category_id: int | None = None,
) -> None:
    if payload.name is not None and not (1 <= len(payload.name) <= 50):
        raise RequestValidationError("Category name must be between 1 and 50 characters long.")
    if payload.context is not None and len(payload.context) > 1000:
        raise RequestValidationError("Category context cannot exceed 1000 characters.")
    if payload.x is not None and not (0 <= payload.x < BOARD_WIDTH):
        raise RequestValidationError(f"X position must be between 0 and {BOARD_WIDTH - 1}.")
    if payload.y is not None and not (0 <= payload.y < BOARD_HEIGHT):
        raise RequestValidationError(f"Y position must be between 0 and {BOARD_HEIGHT - 1}.")
    if payload.width is not None and not (1 <= payload.width <= BOARD_WIDTH):
        raise RequestValidationError(f"Width must be between 1 and {BOARD_WIDTH}.")
    if payload.height is not None and not (1 <= payload.height <= BOARD_HEIGHT):
        raise RequestValidationError(f"Height must be between 1 and {BOARD_HEIGHT}.")
    
    # check if out of bounds
    old = await db.fetch_category(category_id) if category_id is not None else payload
    true_x = payload.x if payload.x is not None else old.x
    true_y = payload.y if payload.y is not None else old.y
    true_width = payload.width if payload.width is not None else old.width
    true_height = payload.height if payload.height is not None else old.height
    
    if true_x + true_width > BOARD_WIDTH or true_y + true_height > BOARD_HEIGHT:
        raise RequestValidationError(
            f"Category position and size must fit within the {BOARD_WIDTH}x{BOARD_HEIGHT} grid."
        )


@app.post('/categories', status_code=201)
async def create_category(
    _: Annotated[Any, AuthorizationRequired()],
    payload: CreateCategoryPayload,
):
    """Create a new category with the provided details."""
    graphic = payload.graphic and base64.b64decode(payload.graphic)
    if graphic and len(graphic) > MAX_GRAPHIC_SIZE:
        raise RequestValidationError(f"Graphic size must not exceed {MAX_GRAPHIC_SIZE} bytes.")

    await validate_category_payload(payload)
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
    await validate_category_payload(payload, category_id=category_id)
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
    payload: list[CreateCategoryPayload],
):
    """Clears all categories and replaces them with the provided list of categories."""
    if not payload:
        raise RequestValidationError("Payload must contain at least one category.")
    for item in payload:
        await validate_category_payload(item)
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
