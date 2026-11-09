import asyncio
from types import SimpleNamespace

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seisbench.data.multiclient import HinetRequest, HinetSource, WaveformRequest

START = UTCDateTime(2010, 5, 1, 3, 0, 20)


def _request(station: str, offset: float, length: float) -> WaveformRequest:
    return WaveformRequest(
        "N", station, "", "EH?", START + offset, START + offset + length
    )


def _source(fetched: list) -> HinetSource:
    source = HinetSource("user", "password", max_span=10)

    def fetch(request: HinetRequest) -> Stream:
        fetched.append(request)
        npts = int((request.endtime - request.starttime) * 100)
        return Stream(
            [
                Trace(
                    np.zeros(npts),
                    header=dict(
                        network="N",
                        station=station,
                        channel=f"EH{component}",
                        starttime=request.starttime,
                        sampling_rate=100.0,
                    ),
                )
                for station in request.stations
                for component in "ZNE"
            ]
        )

    source._fetch = fetch
    return source


def test_hinet_plan_combines_requests():
    fetched = []
    source = _source(fetched)
    # One event at two stations, another 3 minutes after its end, one 1 h later
    requests = [
        _request("AYEH", 0, 120),
        _request("BHRH", 10, 120),
        _request("AYEH", 360, 120),
        _request("RKBH", 3600, 120),
    ]
    source.plan(requests)

    async def get_all():
        return [
            await source.get_waveforms(
                None,
                r.network,
                r.station,
                r.location,
                r.channel,
                r.starttime,
                r.endtime,
            )
            for r in requests
        ]

    streams = asyncio.run(get_all())

    # The first three requests are within 10 minutes and 5 minutes apart: one request
    assert len(fetched) == 2
    first, second = fetched
    assert first.stations == {"AYEH", "BHRH"}
    assert first.starttime == UTCDateTime(2010, 5, 1, 3, 0)
    assert first.endtime == UTCDateTime(2010, 5, 1, 3, 9)
    assert second.stations == {"RKBH"}
    for stream, request in zip(streams, requests):
        assert {tr.stats.station for tr in stream} == {request.station}
        assert len(stream) == 3
        assert stream[0].stats.starttime == request.starttime


def test_hinet_unplanned_request():
    fetched = []
    source = _source(fetched)
    request = _request("AYEH", 0, 60)
    stream = asyncio.run(
        source.get_waveforms(
            None, "N", "AYEH", "", "EH?", request.starttime, request.endtime
        )
    )
    assert len(stream) == 3
    assert fetched == [
        HinetRequest(
            UTCDateTime(2010, 5, 1, 3, 0),
            UTCDateTime(2010, 5, 1, 3, 2),
            frozenset({"AYEH"}),
        )
    ]


def test_hinet_get_stations():
    source = HinetSource("user", "password")
    source._stations = [
        SimpleNamespace(
            code="0101", name="N.AYEH", latitude=35.1, longitude=138.2, elevation=100.0
        ),
        SimpleNamespace(
            code="0101", name="N.BHRH", latitude=36.0, longitude=139.0, elevation=None
        ),
        SimpleNamespace(
            code="0101", name="N.XXXH", latitude=None, longitude=None, elevation=None
        ),
    ]
    inv = asyncio.run(
        source.get_stations(None, "AYEH,BHRH,XXXH,ABC", START, START, "channel")
    )
    assert [station.code for station in inv[0]] == ["AYEH", "BHRH"]
    coordinates = inv.get_coordinates("N.AYEH..EHZ", START)
    assert coordinates["latitude"] == 35.1

    # Other networks are not served
    assert (
        len(
            asyncio.run(
                source.get_stations(None, "AYEH", START, START, "channel", {"AV"})
            )
        )
        == 0
    )
