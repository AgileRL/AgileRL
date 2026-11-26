# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ports a Snowflake job service opens between its instances."""

from __future__ import annotations

import pytest

from agilerl.arena.snowflake.ports import (
    RAY_CLIENT_SERVER_PORT,
    RAY_GCS_PORT,
    RAY_MAX_WORKER_PORT,
    RAY_METRICS_EXPORT_PORT,
    RAY_MIN_WORKER_PORT,
    RAY_PORTS,
    TrainerPorts,
    service_endpoints,
    trainer_port_spans,
)


def _covered(endpoints: list[dict[str, object]]) -> list[int]:
    ports: list[int] = []
    for endpoint in endpoints:
        if "port" in endpoint:
            ports.append(int(str(endpoint["port"])))
        else:
            start, end = str(endpoint["portRange"]).split("-")
            ports.extend(range(int(start), int(end) + 1))
    return ports


class TestRayPorts:
    def test_ray_ports_sit_above_the_ephemeral_range(self) -> None:
        ray_ports = {*RAY_PORTS.values(), RAY_CLIENT_SERVER_PORT}

        assert min(ray_ports) == RAY_GCS_PORT == 61001
        assert max(ray_ports) < RAY_MIN_WORKER_PORT
        assert (RAY_MIN_WORKER_PORT, RAY_MAX_WORKER_PORT) == (61031, 61999)


class TestTrainerPortSpans:
    def test_one_agent_one_engine(self) -> None:
        spans = trainer_port_spans(TrainerPorts(agents=1, engines_per_agent=1))

        assert spans == [(10000, 10031), (20000, 20131), (29700, 29715)]

    def test_spans_grow_with_population_and_engines(self) -> None:
        spans = trainer_port_spans(TrainerPorts(agents=6, engines_per_agent=7))

        last_slot = 5 * 64 + 6
        assert spans == [
            (10000, 10000 + (last_slot + 1) * 32 - 1),
            (20000, 20000 + last_slot * 32 + 131),
            (29700, 29700 + 6 * 16 - 1),
        ]

    def test_largest_population_below_the_ray_ports_is_accepted(self) -> None:
        spans = trainer_port_spans(TrainerPorts(agents=20, engines_per_agent=62))

        assert max(end for _start, end in spans) == 60995

    def test_a_span_that_reaches_the_ray_ports_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reach the Ray ports 61001-61999"):
            trainer_port_spans(TrainerPorts(agents=20, engines_per_agent=63))


class TestServiceEndpoints:
    def test_every_port_is_opened_once(self) -> None:
        trainer = TrainerPorts(agents=6, engines_per_agent=7)

        endpoints = service_endpoints(trainer)
        ports = _covered(endpoints)

        assert len(ports) == len(set(ports))
        wanted = set(RAY_PORTS.values()) | set(range(61031, 62000))
        wanted |= set(range(32768, 61000))
        for start, end in trainer_port_spans(trainer):
            wanted |= set(range(start, end + 1))
        assert set(ports) == wanted

    def test_ray_ports_keep_named_endpoints(self) -> None:
        endpoints = service_endpoints(TrainerPorts(agents=1, engines_per_agent=1))
        by_name = {item["name"]: item for item in endpoints}

        assert by_name["ray-gcs"]["port"] == RAY_GCS_PORT
        assert by_name["ray-metrics"]["port"] == RAY_METRICS_EXPORT_PORT
        assert RAY_CLIENT_SERVER_PORT not in _covered(endpoints)
        assert all(
            item["protocol"] == "TCP" and item["public"] is False for item in endpoints
        )

    def test_overlapping_ranges_are_merged(self) -> None:
        trainer = TrainerPorts(agents=2, engines_per_agent=64)

        endpoints = service_endpoints(trainer)
        ranges = [
            item["portRange"]
            for item in endpoints
            if str(item["name"]).startswith("ports-")
        ]

        assert ranges == [
            "10000-14095",
            "20000-24195",
            "29700-29731",
            "32768-60999",
            "61031-61999",
        ]

    def test_a_population_past_the_ray_ports_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Ray ports"):
            service_endpoints(TrainerPorts(agents=40, engines_per_agent=1))

    def test_no_trainer_omits_trainer_spans(self) -> None:
        endpoints = service_endpoints(None)
        ports = set(_covered(endpoints))

        assert RAY_GCS_PORT in ports
        assert RAY_MIN_WORKER_PORT in ports
        assert 32768 in ports
        assert 10000 not in ports
        assert 20000 not in ports
        assert 29700 not in ports
