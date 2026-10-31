import shutil
import sqlite3
from pathlib import Path

import pytest

from app.config import REPO_ROOT
from app.db import connect_readonly
from app.ingest.catalog import (
    DataValidationError,
    description_is_reliable,
    extract_brand,
    to_cents,
)
from app.ingest.run import build_store

DATA_DIR = REPO_ROOT / "data"


def _one(store_path: Path, query: str, *params) -> sqlite3.Row:
    with connect_readonly(store_path) as connection:
        return connection.execute(query, params).fetchone()


def _product(store_path: Path, product_id: int) -> sqlite3.Row:
    return _one(store_path, "SELECT * FROM products WHERE product_id = ?", product_id)


@pytest.mark.parametrize(
    ("value", "cents"),
    [("799.9", 79990), ("11499", 1149900), ("159.9", 15990), ("0.1", 10)],
)
def test_to_cents_is_exact(value: str, cents: int) -> None:
    assert to_cents(value) == cents


@pytest.mark.parametrize(
    ("name", "brand"),
    [
        ("Music Man Bass 1X Electric Bass", "Music Man"),
        ("Bateria Acústica Pearl Kit 2 Studio", "Pearl"),
        ("Teclado Sintetizador Korg Synth 1 Pro", "Korg"),
        ("Kala KA-15S Soprano Mogno", "Kala"),
        ("Kalani KAL-300S Soprano Natural", "Kalani"),
    ],
)
def test_extract_brand_finds_brand_inside_name(name: str, brand: str) -> None:
    assert extract_brand(name) == brand


def test_extract_brand_rejects_unknown_brand() -> None:
    with pytest.raises(DataValidationError):
        extract_brand("Marca Desconhecida X1")


def test_description_naming_another_brand_is_unreliable() -> None:
    assert not description_is_reliable("Korg", "Sintetizador workstation Roland com 88 teclas")
    assert description_is_reliable("Korg", "Sintetizador Korg com 61 teclas")


def test_every_csv_row_is_loaded(store_path: Path) -> None:
    expected = {
        "categories": 9,
        "products": 65,
        "promotions": 25,
        "customers": 50,
        "orders": 20,
        "order_items": 22,
    }
    for table, count in expected.items():
        assert _one(store_path, f"SELECT COUNT(*) FROM {table}")[0] == count


def test_only_four_promotions_are_active(store_path: Path) -> None:
    with connect_readonly(store_path) as connection:
        rows = connection.execute(
            "SELECT product_id FROM promotions WHERE is_active = 1 ORDER BY product_id"
        ).fetchall()
    assert [row[0] for row in rows] == [90, 94, 121, 127]


def test_string_and_color_values_are_normalized(store_path: Path) -> None:
    steel = _product(store_path, 87)
    assert steel["string_type"] == "aço"
    black_keyboard = _product(store_path, 142)
    portuguese_black = _product(store_path, 89)
    assert black_keyboard["color_family"] == portuguese_black["color_family"] == "preto"


def test_electric_guitar_count_values_are_not_string_types(store_path: Path) -> None:
    guitar = _product(store_path, 145)
    assert guitar["string_type"] is None
    assert guitar["string_count"] == 6
    assert guitar["has_electronics"] == 1


def test_seven_string_guitars_are_identified_by_name(store_path: Path) -> None:
    assert _product(store_path, 112)["string_count"] == 7
    assert _product(store_path, 81)["string_count"] == 6


def test_search_key_ignores_hyphens_and_spaces(store_path: Path) -> None:
    assert _product(store_path, 95)["search_key"].startswith("takaminegd20")


def test_conflicting_descriptions_are_flagged(store_path: Path) -> None:
    with connect_readonly(store_path) as connection:
        rows = connection.execute(
            "SELECT product_id FROM products WHERE description_reliable = 0 ORDER BY product_id"
        ).fetchall()
    assert [row[0] for row in rows] == [135, 137, 139, 140, 142, 144]


def test_recorded_order_total_is_kept_even_when_items_disagree(store_path: Path) -> None:
    order = _one(store_path, "SELECT total_cents FROM orders WHERE order_id = 3")
    assert order["total_cents"] == 345000


def test_report_surfaces_data_oddities(tmp_path: Path) -> None:
    report = build_store(DATA_DIR, tmp_path / "store.db")
    assert report.order_total_mismatches == [3, 20]
    assert report.shared_emails == ["bruno.martins@jmail.com"]
    assert report.unreliable_descriptions == [135, 137, 139, 140, 142, 144]


def test_customer_phones_are_unique_digits(store_path: Path) -> None:
    customer = _one(store_path, "SELECT phone_digits FROM customers WHERE customer_id = 2")
    assert customer["phone_digits"] == "67996547821"


def test_store_is_read_only_for_tools(store_path: Path) -> None:
    with (
        connect_readonly(store_path) as connection,
        pytest.raises(sqlite3.OperationalError),
    ):
        connection.execute("UPDATE products SET price_cents = 1")


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "store.db"
    first = build_store(DATA_DIR, target)
    second = build_store(DATA_DIR, target)
    assert first == second


def test_failed_build_keeps_previous_store(tmp_path: Path) -> None:
    target = tmp_path / "store.db"
    build_store(DATA_DIR, target)

    broken_dir = tmp_path / "broken"
    shutil.copytree(DATA_DIR, broken_dir)
    promotions = broken_dir / "promotions.csv"
    promotions.write_text(
        promotions.read_text(encoding="utf-8").rstrip() + "\n26,127,5,Duplicada,1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError):
        build_store(broken_dir, target)
    assert _one(target, "SELECT COUNT(*) FROM promotions")[0] == 25
