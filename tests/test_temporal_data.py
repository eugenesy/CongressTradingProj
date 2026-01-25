"""
Unit tests for temporal_data.py

Run with: pytest tests/test_temporal_data.py
"""

import pytest
import pandas as pd
import torch
from src.temporal_data import TemporalGraphBuilder


class TestTemporalGraphBuilder:
    """Test suite for TemporalGraphBuilder class"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data for testing"""
        return pd.DataFrame({
            'Ticker': ['AAPL', 'GOOGL', 'AAPL', 'MSFT'],
            'Name': ['Person A', 'Person B', 'Person A', 'Person C'],
            'Filed': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'Traded': pd.to_datetime(['2022-12-28', '2022-12-29', '2022-12-30', '2023-01-01']),
            'Filed_DT': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'Transaction': ['Purchase', 'Sale', 'Purchase', 'Purchase'],
            'Party': ['R', 'D', 'R', 'R'],
            'State': ['CA', 'NY', 'CA', 'TX']
        })
    
    def test_initialization(self, sample_transactions):
        """Test that TemporalGraphBuilder initializes correctly"""
        builder = TemporalGraphBuilder(sample_transactions, min_freq=1)
        assert builder.min_freq == 1
        assert len(builder.transactions) == 4
    
    def test_sorting_by_filed_date(self, sample_transactions):
        """Test that transactions are sorted by Filed date"""
        builder = TemporalGraphBuilder(sample_transactions, min_freq=1)
        filed_dates = builder.transactions['Filed'].tolist()
        assert filed_dates == sorted(filed_dates)
    
    def test_min_frequency_filtering(self):
        """Test that min_freq parameter filters entities correctly"""
        # TODO: Implement test for min_freq filtering
        # This requires creating a dataset where some entities appear < min_freq times
        pass
    
    def test_process_returns_temporal_data(self, sample_transactions):
        """Test that process() returns a TemporalData object"""
        builder = TemporalGraphBuilder(sample_transactions, min_freq=1)
        # TODO: Add price_map fixture and complete this test
        # temporal_data = builder.process(price_map)
        # assert hasattr(temporal_data, 'src')
        # assert hasattr(temporal_data, 'dst')
        # assert hasattr(temporal_data, 't')
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
