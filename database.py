from __future__ import annotations

import asyncio
from typing import Awaitable, TYPE_CHECKING

import aiosqlite

from categorize import Category, CategoryManifest

if TYPE_CHECKING:
    from typing import Self
    
    from server import CreateCategoryPayload

__all__ = ('DbConnection',)


class DbConnection:
    """Manages a connection to the SQL database."""

    DB = 'categories.db'
    SCHEMA = 'schema.sql'
    
    def __init__(self) -> None:
        self._manifest: CategoryManifest | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        self._task: asyncio.Task = self._loop.create_task(self.connect())
        self._cached: dict[int, Category] = {}
    
    async def wait(self) -> Self:
        await self._task
        return self
    
    async def connect(self) -> Self:
        self._inner = await aiosqlite.connect(self.DB, check_same_thread=False)
        self._inner.row_factory = aiosqlite.Row
        self.cursor = await self._inner.cursor()
        await self.initialize_schema()
    
    async def initialize_schema(self) -> None:
        with open(self.SCHEMA, 'r') as schema_file:
            await self.cursor.executescript(schema_file.read())
    
    def commit(self) -> Awaitable[None]:
        """Commit the current transaction."""
        return self._inner.commit()
        
    def close(self) -> Awaitable[None]:
        """Close the database connection."""
        return self._inner.close()

    async def fetch_categories(self, *, respect_cache: bool = True) -> list[Category]:
        if respect_cache and self._cached:
            return list(self._cached.values())

        await self.wait()
        await self.cursor.execute('SELECT * FROM categories')
        rows = await self.cursor.fetchall()
        
        categories = [Category.from_record(row) for row in rows]
        for category in categories:
            self._cached[category.id] = category
        return categories

    async def fetch_category(self, category_id: int, *, respect_cache: bool = True) -> Category:
        """Fetch a single category by its ID."""
        if respect_cache and category_id in self._cached:
            return self._cached[category_id]
        
        await self.wait()
        await self.cursor.execute('SELECT * FROM categories WHERE id = ?', (category_id,))
        row = await self.cursor.fetchone()
        if row is None:
            raise ValueError(f"Category with ID {category_id} does not exist.")
        
        category = Category.from_record(row)
        self._cached[category_id] = category
        return category

    async def create_category(
        self,
        *,
        name: str,
        context: str,
        position: tuple[int, int],
        graphic: bytes | None = None,
        size: tuple[int, int] = (1, 1),
        hue: int = 0,
    ) -> Category:
        """Create a new category in the database."""
        await self.wait()
        await self.cursor.execute(
            'INSERT INTO categories (name, context, x, y, width, height, hue) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (name, context, position[0], position[1], size[0], size[1], hue)
        )
        await self.commit()

        category = Category(
            id=self.cursor.lastrowid,
            position=position,
            name=name,
            context=context,
            size=size,
            hue=hue
        )
        if graphic:
            category.set_graphic(graphic)
            
        self._manifest = None  # Invalidate manifest cache
        self._cached[category.id] = category
        return category

    async def edit_category(
        self,
        category_id: int,
        *,
        name: str | None = None,
        context: str | None = None,
        position: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        hue: int | None = None,
    ) -> Category:
        """Edit an existing category in the database."""
        updates = []
        params = []
        
        if name is not None:
            updates.append('name = ?')
            params.append(name)
        if context is not None:
            updates.append('context = ?')
            params.append(context)
        if position is not None:
            updates.append('x = ?, y = ?')
            params.extend(position)
        if size is not None:
            updates.append('width = ?, height = ?')
            params.extend(size)
        if hue is not None:
            updates.append('hue = ?')
            params.append(hue)
        
        if not updates:
            raise ValueError("No fields to update.")
        
        query = f'UPDATE categories SET {", ".join(updates)} WHERE id = ?'
        params.append(category_id)
        
        await self.wait()
        await self.cursor.execute(query, params)
        await self.commit()
        if self.cursor.rowcount == 0:
            raise KeyError(f"Category with ID {category_id} does not exist.")
        
        if name is not None or context is not None:
            self._manifest = None
        
        return await self.fetch_category(category_id, respect_cache=False)

    async def delete_category(self, category_id: int) -> None:
        """Delete a category by its ID."""
        await self.wait()
        await self.cursor.execute('DELETE FROM categories WHERE id = ?', (category_id,))
        await self.commit()
        
        if self.cursor.rowcount == 0:
            raise KeyError(f"Category with ID {category_id} does not exist.")
        
        self._cached.pop(category_id, None)
        self._manifest = None

    async def clear_categories(self) -> None:
        """Remove all categories from the database."""
        await self.wait()
        await self.cursor.execute('DELETE FROM categories')
        await self.commit()
        self._manifest = None
        self._cached.clear()

    async def set_categories(self, categories: list[CreateCategoryPayload]) -> None:
        """Replace all categories in the database with the provided list."""
        self._cached.clear()
        await self.wait()
        resolved = []
        
        await self.cursor.execute('DELETE FROM categories')
        for category in categories:
            resolved.append(await self.create_category(
                name=category.name,
                context=category.context,
                graphic=category.graphic,
                position=(category.x, category.y),
                size=(category.width, category.height),
                hue=category.hue,
            ))
            
        await self.commit()
        self._manifest = None
        return resolved

    async def fetch_manifest(self) -> CategoryManifest | None:
        if self._manifest is None:
            categories = await self.fetch_categories()
            if not categories:
                return None
            self._manifest = await asyncio.to_thread(
                CategoryManifest.from_categories,
                categories,
            )
            
        return self._manifest
