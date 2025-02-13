import requests
import json
from datetime import datetime
import time


def test_api():
    """
    Test the fraud detection API with sample transactions.
    """
    base_url = "http://localhost:8000"

    # Test health check
    response = requests.get(f"{base_url}/health")
    print("\nHealth Check Response:")
    print(json.dumps(response.json(), indent=2))

    # Sample transactions for testing
    test_transactions = [
        {
            "merchant_id": "MERCH123",
            "amount": 150.00,
            # Add other required fields
        },
        {
            "merchant_id": "MERCH456",
            "amount": 5000.00,
            # Add other required fields
        },
        {
            "merchant_id": "MERCH789",
            "amount": 25.00,
            # Add other required fields
        }
    ]

    print("\nTesting Transaction Predictions:")
    for idx, transaction in enumerate(test_transactions, 1):
        try:
            # Make prediction request
            response = requests.post(
                f"{base_url}/predict",
                json=transaction
            )

            if response.status_code == 200:
                result = response.json()
                print(f"\nTransaction {idx}:")
                print(f"Input: {json.dumps(transaction, indent=2)}")
                print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print(f"\nError for Transaction {idx}:")
                print(f"Status Code: {response.status_code}")
                print(f"Error Message: {response.text}")

        except Exception as e:
            print(f"\nRequest failed for Transaction {idx}:")
            print(f"Error: {str(e)}")

        # Add delay between requests
        time.sleep(1)


if __name__ == "__main__":
    test_api()