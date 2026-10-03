import csv
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from api.app import create_app


def test_stock_persistence_and_alerts(tmp_path):
    path = tmp_path / "products.csv"
    with TestClient(create_app(path)) as client:
        assert client.get("/inventory").json() == []
        response = client.post("/inventory", json={"name": "Oat milk", "quantity": 5, "unit": "liters"})
        assert response.status_code == 201
        product_id = response.json()["id"]
        assert client.get("/inventory/alerts").json()[0]["id"] == product_id
        assert client.patch(f"/inventory/{product_id}", json={"delta": 30}).json()["quantity"] == 35
        assert client.patch(f"/inventory/{product_id}", json={"delta": -12}).json()["quantity"] == 23
        assert client.get("/inventory/alerts").json() == []
        assert len(client.get("/inventory/alerts?threshold=24").json()) == 1
        assert client.get("/inventory/alerts?threshold=23").json() == []
    with TestClient(create_app(path)) as restarted:
        assert restarted.get("/inventory").json()[0]["quantity"] == 23
    with path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_validation_and_locations(tmp_path):
    with TestClient(create_app(tmp_path / "products.csv")) as client:
        product = {"name": "Arabica", "quantity": 10, "unit": "bags", "location": "Downtown"}
        assert client.post("/inventory", json=product).status_code == 201
        assert client.post("/inventory", json={**product, "name": " arabica "}).status_code == 409
        assert client.post("/inventory", json={**product, "location": "Uptown"}).status_code == 201
        assert len(client.get("/inventory?location=downtown").json()) == 1
        assert client.get("/inventory/alerts").json() == []
        assert client.patch("/inventory/1", json={"delta": -11}).status_code == 409
        assert client.get("/inventory").json()[0]["quantity"] == 10
        assert client.patch("/inventory/999", json={"delta": 1}).status_code == 404
        assert client.patch("/inventory/0", json={"delta": 1}).status_code == 422
        for quantity in (-1, "NaN", "Infinity"):
            assert client.post("/inventory", json={**product, "quantity": quantity}).status_code == 422
        assert client.post("/inventory", json={**product, "name": " "}).status_code == 422
        assert client.patch("/inventory/1", json={"delta": "Infinity"}).status_code == 422
        assert client.get("/inventory/alerts?threshold=-1").status_code == 422
        assert client.get("/inventory/alerts?threshold=nan").status_code == 422


def test_concurrent_updates_are_not_lost(tmp_path):
    path = tmp_path / "products.csv"
    with TestClient(create_app(path)) as first, TestClient(create_app(path)) as second:
        first.post("/inventory", json={"name": "Cups", "quantity": 0, "unit": "units"})
        def update(index):
            client = first if index % 2 else second
            return client.patch("/inventory/1", json={"delta": 1}).status_code
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(update, range(30))) == [200] * 30
        assert first.get("/inventory").json()[0]["quantity"] == 30


def test_corrupt_csv_is_not_overwritten(tmp_path):
    path = tmp_path / "products.csv"
    path.write_text("broken,data\n", encoding="utf-8")
    with TestClient(create_app(path)) as client:
        assert client.get("/inventory").status_code == 500
        assert client.post("/inventory", json={"name": "Cups", "quantity": 1, "unit": "units"}).status_code == 500
    assert path.read_text(encoding="utf-8") == "broken,data\n"


def test_fractional_stock_can_be_sold_exactly(tmp_path):
    with TestClient(create_app(tmp_path / "products.csv")) as client:
        client.post("/inventory", json={"name": "Coffee", "quantity": 0.3, "unit": "kg"})
        assert client.patch("/inventory/1", json={"delta": -0.1}).json()["quantity"] == 0.2
        response = client.patch("/inventory/1", json={"delta": -0.2})
        assert response.status_code == 200
        assert response.json()["quantity"] == 0