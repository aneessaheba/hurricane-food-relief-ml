import pytest
import responses
import pandas as pd
from src.data_acquisition import fetch_fema_paginated
from config import API_ENDPOINTS

@responses.activate
def test_fetch_fema_paginated():
    endpoint = API_ENDPOINTS["fema_housing_owners"]
    disaster_number = 9999
    
    # Mock first page
    responses.add(
        responses.GET,
        endpoint,
        json={
            "HousingAssistanceOwners": [
                {"zipCode": "32003", "totalInspected": 10},
                {"zipCode": "32004", "totalInspected": 5}
            ]
        },
        status=200
    )
    
    # Mock second page (empty to stop pagination)
    responses.add(
        responses.GET,
        endpoint,
        json={"HousingAssistanceOwners": []},
        status=200
    )
    
    df = fetch_fema_paginated(endpoint, disaster_number, top=2, max_pages=5)
    
    # Assert it grabbed 2 rows
    assert len(df) == 2
    assert "zipCode" in df.columns
    assert df.iloc[0]["zipCode"] == "32003"
