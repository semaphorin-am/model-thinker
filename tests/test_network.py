"""Tests for network module."""

import pytest
import numpy as np

from congestion_pricing.network.graph import RoadNetwork, NetworkBuilder


class TestRoadNetwork:
    """Tests for RoadNetwork class."""

    def test_create_empty_network(self):
        """Test creating an empty network."""
        network = RoadNetwork()

        assert network.n_nodes == 0
        assert network.n_edges == 0

    def test_add_node(self):
        """Test adding nodes."""
        network = RoadNetwork()
        network.add_node("A", x=0.0, y=0.0)
        network.add_node("B", x=1.0, y=0.0)

        assert network.n_nodes == 2

    def test_add_edge(self):
        """Test adding edges."""
        network = RoadNetwork()
        network.add_node("A", x=0.0, y=0.0)
        network.add_node("B", x=1.0, y=0.0)
        network.add_edge("A", "B", length=1.0, capacity=1000, free_flow_time=1.0)

        assert network.n_edges == 1

    def test_bpr_time(self):
        """Test BPR travel time function."""
        network = RoadNetwork()
        network.add_node("A", x=0.0, y=0.0)
        network.add_node("B", x=1.0, y=0.0)
        network.add_edge("A", "B", length=1.0, capacity=1000, free_flow_time=10.0)

        # No flow - should equal free flow time
        time_empty = network.get_travel_time("A", "B", flow=0)
        assert time_empty == 10.0

        # At capacity - should be higher
        time_at_capacity = network.get_travel_time("A", "B", flow=1000)
        assert time_at_capacity > 10.0

    def test_shortest_path(self):
        """Test shortest path finding."""
        network = RoadNetwork()
        network.add_node("A", x=0.0, y=0.0)
        network.add_node("B", x=1.0, y=0.0)
        network.add_node("C", x=2.0, y=0.0)
        network.add_edge("A", "B", length=1.0, capacity=1000, free_flow_time=1.0)
        network.add_edge("B", "C", length=1.0, capacity=1000, free_flow_time=1.0)
        network.add_edge("A", "C", length=3.0, capacity=1000, free_flow_time=3.0)

        path = network.shortest_path("A", "C")

        # Should go through B (total time 2) not direct (time 3)
        assert path == ["A", "B", "C"]


class TestNetworkBuilder:
    """Tests for NetworkBuilder class."""

    def test_create_grid_network(self):
        """Test creating a grid network."""
        builder = NetworkBuilder()
        network = builder.create_grid(rows=3, cols=3, spacing=100.0)

        assert network.n_nodes == 9
        # Each internal node has 4 edges, edge nodes have 2-3
        assert network.n_edges > 0

    def test_create_test_network(self):
        """Test creating the Braess-style test network."""
        builder = NetworkBuilder()
        network = builder.create_test_network()

        # Should have O, A, B, D nodes
        assert network.n_nodes >= 4
        assert network.n_edges >= 4


class TestNetworkCapacity:
    """Tests for capacity functions."""

    def test_flow_below_capacity(self):
        """Test that flow below capacity has low delay."""
        from congestion_pricing.network.capacity import bpr_travel_time

        travel_time = bpr_travel_time(
            free_flow_time=10.0,
            flow=500,
            capacity=1000,
        )

        # Should be close to free flow time
        assert travel_time < 11.0

    def test_flow_at_capacity(self):
        """Test delay at capacity."""
        from congestion_pricing.network.capacity import bpr_travel_time

        travel_time = bpr_travel_time(
            free_flow_time=10.0,
            flow=1000,
            capacity=1000,
            alpha=0.15,
            beta=4.0,
        )

        # BPR: t = t0 * (1 + 0.15 * 1^4) = 10 * 1.15 = 11.5
        np.testing.assert_almost_equal(travel_time, 11.5)

    def test_flow_over_capacity(self):
        """Test significant delay over capacity."""
        from congestion_pricing.network.capacity import bpr_travel_time

        travel_time = bpr_travel_time(
            free_flow_time=10.0,
            flow=1500,
            capacity=1000,
        )

        # Should be significantly higher
        assert travel_time > 15.0
