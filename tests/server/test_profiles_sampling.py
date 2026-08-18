"""A model's own sampling row: the level under every profile of it, on its own sub-resource.

The stand is `profile_stand.py`'s, and the profiles above this level are
`test_profiles.py`'s.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia.server.services import catalog

from .conftest import wired
from .profile_stand import (
    ECHO,
    MODEL,
    SCANNER,
    TINY,
    Loader,
    ask,
    echo,
    installed,
    put,
    sample,
    settings_row,
    tiny,
)


@pytest.fixture
def loader() -> Loader:
    return Loader({TINY: tiny(), ECHO: echo()})


@pytest.fixture
def client(loader: Loader) -> Iterator[TestClient]:
    with TestClient(wired(loader)) as running:
        yield running


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The scan reads two module constants; the user's own caches are never touched."""
    root = tmp_path / "hub"
    monkeypatch.setattr(SCANNER, "HUB_CACHE", root)
    monkeypatch.setattr(SCANNER, "QUANTIZED_CACHE", tmp_path / "quantized")
    catalog.context_of.cache_clear()
    catalog.defaults_of.cache_clear()
    return root


def test_a_models_sampling_is_written_read_and_replaced(client: TestClient) -> None:
    """The level under every profile of a model, on its own sub-resource. A `PUT` replaces
    rather than patches, like a profile's: what the second body omits is a knob this daemon
    no longer opines on, which is the only reading under which the body is the setting."""
    url = f"/admin/models/{MODEL}/sampling"
    assert client.get(url).json()["temperature"] is None

    written = client.put(url, json={"temperature": 0.2, "top_k": 5})
    assert written.status_code == 200, written.text
    assert client.get(url).json() == written.json()
    assert json.loads(settings_row(MODEL)[0]) == {"temperature": 0.2, "top_k": 5}

    assert client.put(url, json={"seed": 3}).status_code == 200
    replaced = client.get(url).json()
    assert replaced["seed"] == 3
    assert replaced["temperature"] is None


def test_a_models_sampling_wins_over_what_the_checkpoint_declares(
    client: TestClient, hub: Path
) -> None:
    """`generation_config.json` says how the people who trained it meant it to be sampled;
    this row says how the person running it wants it sampled here, and the second is the more
    specific of the two. Seeded, so the answer repeating says the row was read and the
    greedy answer it is not says the file was outranked."""
    installed(hub, TINY, generation={"do_sample": False, "temperature": 0.6})

    greedy = ask(client, TINY)
    sample(client, TINY, temperature=2, seed=7)
    hot = ask(client, TINY)
    assert hot != greedy
    assert hot == ask(client, TINY)


def test_a_profile_wins_over_the_models_sampling_and_fills_only_what_it_left_unset(
    client: TestClient,
) -> None:
    """Two levels between the request and the checkpoint, and the profile is the upper one:
    a preset written for a job outranks what the model says in general. What the profile does
    not name is not overridden, though — the seed below is the model's, and it is what makes
    the sampled answer repeat."""
    sample(client, TINY, temperature=2, seed=7)
    put(client, TINY, "cold", sampling={"temperature": 0})
    put(client, TINY, "hot", sampling={"top_k": 3})

    cold = ask(client, f"{TINY}:cold")
    assert cold == ask(client, f"{TINY}:cold")
    hot = ask(client, f"{TINY}:hot")
    assert hot != cold
    assert hot == ask(client, f"{TINY}:hot"), "the seed the profile never named is the model's"


def test_a_knob_the_request_sends_wins_over_the_models_sampling(client: TestClient) -> None:
    """The top of the four levels is still the request: a client that names a temperature
    means it, whatever this daemon was told about the model."""
    sample(client, TINY, temperature=2, seed=7)

    hot = ask(client, TINY)
    greedy = ask(client, TINY, temperature=0)
    assert greedy != hot
    assert greedy == ask(client, TINY, temperature=0)


def test_a_models_sampling_and_its_features_are_kept_in_the_same_row_without_erasing_each_other(
    client: TestClient, hub: Path
) -> None:
    """Two routes write one row. A settings `PUT` that dropped the sampling — or a sampling
    `PUT` that dropped the batch limit — would be a switch the reader set and cannot see."""
    installed(hub, TINY)
    sample(client, TINY, temperature=0.3)
    settings = client.put(
        f"/admin/models/{TINY}/settings", json={"features": {}, "max_concurrent_requests": 2}
    )
    assert settings.status_code == 200, settings.text
    assert client.get(f"/admin/models/{TINY}/sampling").json()["temperature"] == 0.3

    sample(client, TINY, temperature=0.4)
    assert settings_row(TINY)[1] == 2


def test_a_profile_called_sampling_is_not_read_as_the_models_own_row(client: TestClient) -> None:
    """`{model_id:path}` matches slashes, so the model's sampling route registered above the
    profiles would answer this `PUT` as the model `test/tiny/profiles`. The name is not
    reserved and nothing says it should be — what says the order is right is the profile
    coming back from its own `GET`."""
    put(client, TINY, "sampling", sampling={"temperature": 0.5})

    assert client.get(f"/admin/models/{TINY}/profiles/sampling").json()["name"] == "sampling"
    assert client.get(f"/admin/models/{TINY}/sampling").json()["temperature"] is None
    assert settings_row(f"{TINY}/profiles")[0] == "{}"
