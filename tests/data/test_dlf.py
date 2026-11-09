import asyncio

from obspy import UTCDateTime
from obspy.core.event import (
    Comment,
    Event,
    EventDescription,
    Magnitude,
    Origin,
    ResourceIdentifier,
)

from seisbench.data.dlf import get_event_params, is_quiet
from seisbench.data.utils import FDSNSource


def test_is_quiet():
    origins = [UTCDateTime(2020, 1, 1, 0, 10).timestamp]
    event = UTCDateTime(2020, 1, 1, 0, 10)

    # Ends exactly 120 s before the event
    assert is_quiet(origins, event - 180, event - 120)
    assert not is_quiet(origins, event - 179, event - 119)
    # Starts exactly 180 s after the event
    assert is_quiet(origins, event + 180, event + 240)
    assert not is_quiet(origins, event + 179, event + 239)
    assert not is_quiet(origins, event - 30, event + 30)
    assert is_quiet([], event, event + 60)


def test_get_event_params():
    origin = Origin(
        time=UTCDateTime(2015, 3, 1),
        latitude=19.2,
        longitude=-155.4,
        depth=40e3,
    )
    magnitude = Magnitude(mag=1.2)
    event = Event(
        resource_id=ResourceIdentifier(
            "smi:local/phasenet_dlf_v4b/hawaii/hawaii2_60234378"
        ),
        origins=[origin],
        magnitudes=[magnitude],
        comments=[Comment(text="split: test"), Comment(text="event_class: DLP")],
        event_descriptions=[
            EventDescription(text="Pahala deep swarm", type="region name")
        ],
    )
    event.preferred_origin_id = origin.resource_id
    event.preferred_magnitude_id = magnitude.resource_id

    params = get_event_params(event, "hawaii")
    assert params["source_id"] == "hawaii:hawaii2_60234378"
    assert params["split"] == "test"
    assert params["source_event_class"] == "DLP"
    assert params["source_volcano"] == "Pahala deep swarm"
    assert params["source_region"] == "hawaii"
    assert params["source_depth_km"] == 40.0
    assert params["source_magnitude"] == 1.2


def test_fdsn_source_networks():
    source = FDSNSource("https://example.org", networks=["NC", "BK"])

    async def get_stations(networks):
        # Returns before any request if the source serves none of the networks
        return await source.get_stations(
            None, "STA", UTCDateTime(0), UTCDateTime(1), "station", networks
        )

    assert len(asyncio.run(get_stations({"AV", "HV"}))) == 0
