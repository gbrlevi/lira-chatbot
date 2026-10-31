"""Loads the store CSVs into SQLite, fixing what would otherwise mislead the agent."""

import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from app.text import compact, digits_only, normalize

CATEGORY_GUITARS = 1
CATEGORY_BASSES = 2
CATEGORY_ACOUSTIC_GUITARS = 5
CATEGORY_UKULELES = 9

# Brands are not a column in the source data and must be recognized inside product names.
# A naive first-word split would produce "Bateria", "Teclado" and "Music".
PRODUCT_BRANDS = (
    "Music Man", "Yamaha", "Kala", "Kalani", "Tagima", "Giannini", "Shelby", "Takamine",
    "Ohana", "Rozini", "Crafter", "Martin", "Taylor", "Fender", "Ibanez", "Gibson", "PRS",
    "Pearl", "Korg", "Nord", "Roland",
)  # fmt: skip
# Brands the store does not carry but that appear inside inconsistent descriptions.
OTHER_BRANDS = ("Tama",)

STRING_TYPES = {"nylon": "nylon", "steel": "aço"}

# Each stored color maps to the family a customer would ask for.
COLOR_FAMILIES = {
    "natural": "natural",
    "sunburst": "sunburst",
    "heritage cherry sunburst": "sunburst",
    "black": "preto",
    "preto": "preto",
    "red": "vermelho",
    "wine red": "vinho",
    "blue": "azul",
    "desert sun yellow": "amarelo",
    "eriza verde": "verde",
    "butterscotch blonde": "amarelo",
}

SCHEMA = """
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    search_key TEXT NOT NULL,
    brand TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories (category_id),
    description TEXT NOT NULL,
    description_reliable INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    stock_quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    string_type TEXT,
    string_count INTEGER,
    has_electronics INTEGER,
    color TEXT,
    color_family TEXT,
    specs TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE promotions (
    promotion_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    discount_percent INTEGER NOT NULL,
    description TEXT NOT NULL,
    is_active INTEGER NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone_digits TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    city TEXT NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total_cents INTEGER NOT NULL,
    payment_method TEXT NOT NULL,
    tracking_code TEXT,
    estimated_delivery TEXT,
    notes TEXT
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders (order_id),
    product_id INTEGER NOT NULL REFERENCES products (product_id),
    quantity INTEGER NOT NULL
);
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_customers_email ON customers (email);
CREATE INDEX idx_items_order ON order_items (order_id);
CREATE INDEX idx_promotions_product ON promotions (product_id);
"""


class DataValidationError(ValueError):
    """The source data breaks a rule the agent depends on."""


@dataclass(frozen=True)
class CatalogReport:
    """Source data oddities that are handled rather than rejected."""

    products: int
    unreliable_descriptions: list[int] = field(default_factory=list)
    order_total_mismatches: list[int] = field(default_factory=list)
    shared_emails: list[str] = field(default_factory=list)


def to_cents(value: str) -> int:
    return int((Decimal(value) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _mentions(brand: str, normalized_text: str) -> bool:
    return re.search(rf"\b{re.escape(normalize(brand))}\b", normalized_text) is not None


def extract_brand(name: str) -> str:
    normalized_name = normalize(name)
    for brand in PRODUCT_BRANDS:
        if _mentions(brand, normalized_name):
            return brand
    raise DataValidationError(f"Marca não reconhecida no produto: {name!r}")


def description_is_reliable(brand: str, description: str) -> bool:
    """A description naming another brand contradicts the product it belongs to."""
    normalized = normalize(description)
    others = (other for other in (*PRODUCT_BRANDS, *OTHER_BRANDS) if other != brand)
    return not any(_mentions(other, normalized) for other in others)


def extract_string_count(name: str, category_id: int, specs: dict) -> int | None:
    if re.search(r"\b7 cordas\b", normalize(name)):
        return 7
    raw = str(specs.get("strings", ""))
    if raw.isdigit():
        return int(raw)
    defaults = {CATEGORY_ACOUSTIC_GUITARS: 6, CATEGORY_UKULELES: 4}
    return defaults.get(category_id)


def extract_has_electronics(category_id: int, specs: dict) -> bool | None:
    if category_id in (CATEGORY_GUITARS, CATEGORY_BASSES):
        return True
    value = specs.get("electronics")
    return None if value is None else value == "yes"


def normalize_color(specs: dict) -> tuple[str | None, str | None]:
    color = specs.get("color")
    if color is None:
        return None, None
    family = COLOR_FAMILIES.get(normalize(color))
    if family is None:
        raise DataValidationError(f"Cor sem família definida: {color!r}")
    return color, family


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _product_row(raw: dict[str, str]) -> tuple:
    category_id = int(raw["category_id"])
    specs = json.loads(raw["specs"])
    brand = extract_brand(raw["name"])
    color, color_family = normalize_color(specs)
    has_electronics = extract_has_electronics(category_id, specs)
    return (
        int(raw["product_id"]),
        raw["name"],
        compact(raw["name"]),
        brand,
        category_id,
        raw["description"],
        int(description_is_reliable(brand, raw["description"])),
        to_cents(raw["price_brl"]),
        int(raw["stock_quantity"]),
        raw["status"],
        STRING_TYPES.get(str(specs.get("strings"))),
        extract_string_count(raw["name"], category_id, specs),
        None if has_electronics is None else int(has_electronics),
        color,
        color_family,
        raw["specs"],
        raw["created_at"],
    )


def _validate(connection: sqlite3.Connection) -> None:
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise DataValidationError(f"Referências quebradas nos dados: {len(violations)}")
    repeated = connection.execute(
        "SELECT product_id FROM promotions WHERE is_active = 1"
        " GROUP BY product_id HAVING COUNT(*) > 1"
    ).fetchall()
    if repeated:
        ids = [row[0] for row in repeated]
        raise DataValidationError(f"Produtos com mais de uma promoção ativa: {ids}")


def _build_report(connection: sqlite3.Connection) -> CatalogReport:
    products = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    unreliable = connection.execute(
        "SELECT product_id FROM products WHERE description_reliable = 0 ORDER BY product_id"
    ).fetchall()
    mismatches = connection.execute(
        """
        SELECT o.order_id FROM orders o
        JOIN order_items i ON i.order_id = o.order_id
        JOIN products p ON p.product_id = i.product_id
        GROUP BY o.order_id
        HAVING o.total_cents <> SUM(p.price_cents * i.quantity)
        ORDER BY o.order_id
        """
    ).fetchall()
    shared = connection.execute(
        "SELECT email FROM customers GROUP BY email HAVING COUNT(*) > 1 ORDER BY email"
    ).fetchall()
    return CatalogReport(
        products=products,
        unreliable_descriptions=[row[0] for row in unreliable],
        order_total_mismatches=[row[0] for row in mismatches],
        shared_emails=[row[0] for row in shared],
    )


def load_catalog(connection: sqlite3.Connection, data_dir: Path) -> CatalogReport:
    """Create the schema and load every CSV. Expects an empty database."""
    connection.executescript(SCHEMA)
    connection.executemany(
        "INSERT INTO categories VALUES (?, ?, ?)",
        [
            (int(row["category_id"]), row["name"], row["description"])
            for row in _read_csv(data_dir / "categories.csv")
        ],
    )
    connection.executemany(
        f"INSERT INTO products VALUES ({', '.join('?' * 17)})",
        [_product_row(row) for row in _read_csv(data_dir / "products.csv")],
    )
    connection.executemany(
        "INSERT INTO promotions VALUES (?, ?, ?, ?, ?)",
        [
            (
                int(row["promotion_id"]),
                int(row["product_id"]),
                int(row["discount_percent"]),
                row["description"],
                int(row["is_active"]),
            )
            for row in _read_csv(data_dir / "promotions.csv")
        ],
    )
    connection.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (
                int(row["customer_id"]),
                row["name"],
                digits_only(row["phone"]),
                row["email"].strip().lower(),
                row["city"],
            )
            for row in _read_csv(data_dir / "customers.csv")
        ],
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                int(row["order_id"]),
                int(row["customer_id"]),
                row["order_date"],
                row["status"],
                to_cents(row["total_brl"]),
                row["payment_method"],
                row["tracking_code"] or None,
                row["estimated_delivery"] or None,
                row["notes"] or None,
            )
            for row in _read_csv(data_dir / "orders.csv")
        ],
    )
    connection.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?)",
        [
            (int(row["order_id"]), int(row["product_id"]), int(row["quantity"]))
            for row in _read_csv(data_dir / "order_items.csv")
        ],
    )
    _validate(connection)
    return _build_report(connection)
