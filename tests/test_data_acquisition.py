import pytest
import requests
import responses
from urllib.parse import parse_qs, urlparse

from src.data_acquisition import (
    download_file, fetch_fema_paginated, request_with_retry, _backoff_seconds,
)
from config import API_ENDPOINTS

NO_SLEEP = lambda s: None  # noqa: E731


def _filter_of(call):
    return parse_qs(urlparse(call.request.url).query)["$filter"][0]


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
                {"id": "00000000-0000-0000-0000-000000000001", "zipCode": "32003", "totalInspected": 10},
                {"id": "00000000-0000-0000-0000-000000000002", "zipCode": "32004", "totalInspected": 5}
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


@responses.activate
def test_fema_keyset_paging_quotes_uuid_ids():
    endpoint = API_ENDPOINTS["fema_housing_owners"]
    uuid = "00173a0c-6663-4011-98e8-803379d5ba9b"
    responses.add(responses.GET, endpoint,
                  json={"HousingAssistanceOwners": [{"id": "0012c98d-0000-0000-0000-000000000000"},
                                                    {"id": uuid}]})
    responses.add(responses.GET, endpoint,
                  json={"HousingAssistanceOwners": [{"id": "ffff0000-0000-0000-0000-000000000000"}]})

    df = fetch_fema_paginated(endpoint, 4834, top=2, sleep=NO_SLEEP)

    assert len(df) == 3
    first, second = responses.calls
    assert _filter_of(first) == "disasterNumber eq 4834"
    assert _filter_of(second) == f"disasterNumber eq 4834 and id gt '{uuid}'"
    assert parse_qs(urlparse(second.request.url).query)["$orderby"] == ["id"]
    # Keyset paging replaces deep $skip offsets.
    assert "$skip" not in parse_qs(urlparse(second.request.url).query)


@responses.activate
def test_fema_keyset_paging_leaves_integer_ids_bare():
    endpoint = API_ENDPOINTS["fema_ihp_registrations"]
    key = "IndividualsAndHouseholdsProgramValidRegistrations"
    responses.add(responses.GET, endpoint, json={key: [{"id": 10}, {"id": 11}]})
    responses.add(responses.GET, endpoint, json={key: []})

    fetch_fema_paginated(endpoint, 4611, top=2, sleep=NO_SLEEP)

    assert _filter_of(responses.calls[1]) == "disasterNumber eq 4611 and id gt 11"


@responses.activate
def test_fema_paging_retries_transient_errors():
    endpoint = API_ENDPOINTS["fema_housing_renters"]
    responses.add(responses.GET, endpoint, status=503)
    responses.add(responses.GET, endpoint, body=requests.ConnectionError("reset"))
    responses.add(responses.GET, endpoint, json={"HousingAssistanceRenters": [{"id": "a"}]})

    df = fetch_fema_paginated(endpoint, 4830, top=10, sleep=NO_SLEEP)

    assert len(df) == 1
    assert len(responses.calls) == 3


@responses.activate
def test_fema_paging_raises_on_api_error_payload():
    endpoint = API_ENDPOINTS["fema_housing_owners"]
    responses.add(responses.GET, endpoint,
                  json={"error": [{"message": "Invalid UUID"}]})

    with pytest.raises(RuntimeError, match="FEMA API error"):
        fetch_fema_paginated(endpoint, 4834, sleep=NO_SLEEP)


@responses.activate
def test_request_with_retry_gives_up_after_max_retries():
    url = "https://example.test/data"
    responses.add(responses.GET, url, status=502)
    sleeps = []

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        request_with_retry(url, max_retries=3, sleep=sleeps.append)

    assert len(responses.calls) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


@responses.activate
def test_request_with_retry_does_not_retry_client_errors():
    url = "https://example.test/missing"
    responses.add(responses.GET, url, status=404)

    with pytest.raises(requests.HTTPError):
        request_with_retry(url, max_retries=5, sleep=NO_SLEEP)

    assert len(responses.calls) == 1


@responses.activate
def test_request_with_retry_honours_retry_after():
    url = "https://example.test/limited"
    responses.add(responses.GET, url, status=429, headers={"Retry-After": "7"})
    responses.add(responses.GET, url, body="ok")
    sleeps = []

    assert request_with_retry(url, sleep=sleeps.append).text == "ok"
    assert sleeps == [7.0]


def test_backoff_is_capped_and_jittered():
    for attempt in range(12):
        wait = _backoff_seconds(attempt)
        assert 0 < wait <= 60
    assert _backoff_seconds(0, retry_after="not-a-number") <= 1


@responses.activate
def test_download_file_writes_and_skips_existing(tmp_path):
    url = "https://example.test/file.csv"
    responses.add(responses.GET, url, body=b"a,b\n1,2\n")
    dest = tmp_path / "file.csv"

    assert download_file(url, dest) == dest
    assert dest.read_bytes() == b"a,b\n1,2\n"
    assert not (tmp_path / "file.csv.part").exists()

    download_file(url, dest)  # existing file: no second request
    assert len(responses.calls) == 1


@responses.activate
def test_download_file_leaves_no_partial_file_on_failure(tmp_path):
    url = "https://example.test/gone.csv"
    responses.add(responses.GET, url, status=404)
    dest = tmp_path / "gone.csv"

    with pytest.raises(RuntimeError, match="Download failed"):
        download_file(url, dest)

    assert not dest.exists()
    assert not (tmp_path / "gone.csv.part").exists()
