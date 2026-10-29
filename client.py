"""Python client that tests the FastAPI prediction endpoint."""
import requests

API_URL = "http://127.0.0.1:8000"

def main():
    passenger = {
        "Pclass": 3,
        "Sex": "female",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        print("Health status:", health.status_code, health.json())

        response = requests.post(
            f"{API_URL}/predict", json=passenger, timeout=10
        )
        print("Prediction status:", response.status_code)
        print("Prediction response:", response.json())
        response.raise_for_status()
    except requests.RequestException as exc:
        print("Could not reach the API. Start it with: uvicorn app:app --reload")
        print(f"Details: {exc}")

if __name__ == "__main__":
    main()
