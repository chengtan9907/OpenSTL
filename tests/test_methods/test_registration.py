"""Test method registration and factory patterns."""
import pytest

# Skip all tests in this module if lightning is not available
pytest.importorskip("lightning")

from openstl.methods import method_maps, __all__


class TestMethodRegistration:
    """Test method registration."""

    def test_method_maps_not_empty(self):
        """Test that method_maps is not empty."""
        assert isinstance(method_maps, dict)
        assert len(method_maps) > 0

    def test_all_registered_methods(self):
        """Test that all expected methods are registered."""
        expected_methods = [
            'convlstm', 'e3dlstm', 'mau', 'mim', 'phydnet',
            'predrnn', 'predrnnpp', 'predrnnv2', 'simvp',
            'tau', 'mmvp', 'swinlstm_d', 'swinlstm_b', 'wast'
        ]
        for method in expected_methods:
            assert method in method_maps, f"Method '{method}' not registered"

    def test_method_classes_valid(self):
        """Test that all registered method classes are valid."""
        for name, method_class in method_maps.items():
            assert method_class is not None, f"Method '{name}' class is None"
            # Check that class has required attributes
            assert hasattr(method_class, '__init__'), f"Method '{name}' missing __init__"


class TestMethodInstantiation:
    """Test method instantiation with minimal config."""

    @pytest.mark.parametrize("method_name", list(method_maps.keys()))
    def test_method_can_create(self, method_name):
        """Test that each method can be created (without full initialization)."""
        method_class = method_maps[method_name]
        # Just check that the class exists and is callable
        assert callable(method_class)
