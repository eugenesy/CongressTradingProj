"""
Unit tests for models_tgn.py

Run with: pytest tests/test_models.py
"""

import pytest
import torch
from src.models_tgn import TGN, GraphAttentionEmbedding, PriceEncoder


class TestPriceEncoder:
    """Test suite for PriceEncoder"""
    
    def test_price_encoder_forward(self):
        """Test PriceEncoder forward pass"""
        encoder = PriceEncoder(input_dim=14, hidden_dim=32, output_dim=32)
        
        # Create dummy input (batch_size=10, features=14)
        x = torch.randn(10, 14)
        
        output = encoder(x)
        
        assert output.shape == (10, 32), f"Expected shape (10, 32), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestTGN:
    """Test suite for TGN model"""
    
    @pytest.fixture
    def model_config(self):
        """Default model configuration"""
        return {
            'memory_dim': 100,
            'time_dim': 100,
            'embedding_dim': 100,
            'num_pol_features': 16,
            'price_feature_dim': 32,
            'num_heads': 4,
            'dropout': 0.1
        }
    
    def test_model_initialization(self, model_config):
        """Test that TGN model initializes without errors"""
        model = TGN(**model_config)
        assert model is not None
        assert model.memory_dim == 100
    
    def test_model_forward_shape(self, model_config):
        """Test that forward pass produces correct output shape"""
        # TODO: Implement forward pass test with dummy data
        # This requires creating mock temporal graph data
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
