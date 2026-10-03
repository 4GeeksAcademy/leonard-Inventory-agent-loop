import csv
import math
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path as APIPath, Query
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["id", "name", "quantity", "unit", "location"]


class NewProduct(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    quantity: FiniteFloat = Field(ge=0)
    unit: str = Field(min_length=1, max_length=40)
    location: str = Field(default="Main", min_length=1, max_length=80)


class Product(NewProduct):
    id: int = Field(gt=0)


class StockChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: FiniteFloat


class InventoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(path) + ".lock")

    def read(self) -> list[Product]:
        if not self.path.exists():
            self.write([])
        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Invalid inventory CSV header")
                products = [Product.model_validate(row) for row in reader]
            if len({product.id for product in products}) != len(products):
                raise ValueError("Duplicate product IDs")
            return products
        except (ValueError, TypeError) as error:
            raise HTTPException(500, "Inventory CSV is invalid; repair it before continuing.") from error

    def write(self, products: list[Product]) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=self.path.parent,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(product.model_dump() for product in products)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def create_app(products_path: Path | None = None) -> FastAPI:
    application = FastAPI(title="Carla's Inventory API", version="1.0.0")
    store = InventoryStore(products_path or Path(os.getenv("PRODUCTS_FILE", str(ROOT / "products.csv"))))
    default_threshold = float(os.getenv("LOW_STOCK_THRESHOLD", "10"))
    if not math.isfinite(default_threshold) or default_threshold < 0:
        raise ValueError("LOW_STOCK_THRESHOLD must be finite and nonnegative")

    @application.get("/inventory", response_model=list[Product])
    def inventory(location: str | None = None):
        with store.lock:
            products = store.read()
        return [product for product in products if location is None or product.location.casefold() == location.strip().casefold()]

    @application.post("/inventory", response_model=Product, status_code=201)
    def add_product(product: NewProduct):
        with store.lock:
            products = store.read()
            if any(
                existing.name.casefold() == product.name.casefold()
                and existing.location.casefold() == product.location.casefold()
                for existing in products
            ):
                raise HTTPException(409, "That product already exists at this location; update its stock instead.")
            created = Product(id=max((existing.id for existing in products), default=0) + 1, **product.model_dump())
            products.append(created)
            store.write(products)
        return created

    @application.get("/inventory/alerts", response_model=list[Product])
    def alerts(
        threshold: Annotated[float, Query(ge=0, allow_inf_nan=False)] = default_threshold,
        location: str | None = None,
    ):
        return [product for product in inventory(location) if product.quantity < threshold]

    @application.patch("/inventory/{product_id}", response_model=Product)
    def update_stock(product_id: Annotated[int, APIPath(gt=0)], change: StockChange):
        with store.lock:
            products = store.read()
            product = next((existing for existing in products if existing.id == product_id), None)
            if product is None:
                raise HTTPException(404, f"Product {product_id} was not found.")
            quantity = float(Decimal(str(product.quantity)) + Decimal(str(change.delta)))
            if quantity < 0:
                raise HTTPException(409, f"Insufficient stock: only {product.quantity:g} {product.unit} available.")
            if not math.isfinite(quantity):
                raise HTTPException(422, "The resulting quantity must be finite.")
            product.quantity = quantity
            store.write(products)
        return product

    return application


app = create_app()