
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_find_similar_products():

    product_id = (
        app.state.engine.dataframe.iloc[0]["uniq_id"]
    )

    response = client.post(
        "/find_similar_products",
        json={
            "product_id": product_id,
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    print(data)
    print("\nReturned Keys:")
    print(data.keys())

    print("\nNumber of Results:")
    print(len(data["results"]))

    print("\nFirst Result:")
    print(data["results"][0])

    assert len(data["results"]) == 5