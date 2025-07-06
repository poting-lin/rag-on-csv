"""
Pytest configuration for CSV QA Bot integration tests.
"""
import pytest
import os
import pandas as pd


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def test_data_dir(project_root):
    """Return the path to the test data directory."""
    return os.path.join(project_root, "tests", "test_data")


@pytest.fixture
def sound_measurements_csv(test_data_dir):
    """Return the path to the sound measurements CSV file."""
    return os.path.join(test_data_dir, "sound_measurements.csv")


@pytest.fixture
def lab_data_csv(test_data_dir):
    """Return the path to the lab data CSV file."""
    return os.path.join(test_data_dir, "lab_data.csv")


@pytest.fixture
def bookshop_csv(test_data_dir):
    """Return the path to the bookshop CSV file."""
    return os.path.join(test_data_dir, "bookshop.csv")


@pytest.fixture
def sample_data_dir(project_root):
    """Return the path to the sample data directory."""
    return os.path.join(project_root, "sample_data")


@pytest.fixture
def sound_measurements_path(project_root):
    """Return the path to the sound measurements CSV file"""
    return os.path.join(project_root, "sample_data", "sound_measurements.csv")


@pytest.fixture
def lab_data_path(project_root):
    """Return the path to the lab data CSV file"""
    return os.path.join(project_root, "sample_data", "lab_data.csv")


@pytest.fixture
def bookshop_path(project_root):
    """Return the path to the bookshop CSV file"""
    return os.path.join(project_root, "sample_data", "bookshop.csv")


@pytest.fixture
def sound_measurements_df(sound_measurements_path):
    """Load the sound measurements CSV as a pandas DataFrame"""
    return pd.read_csv(sound_measurements_path)


@pytest.fixture
def event_records(sound_measurements_df):
    """Return records with EventType = Event"""
    return sound_measurements_df[sound_measurements_df["EventType"] == "Event"]


@pytest.fixture
def high_decibel_records(sound_measurements_df):
    """Return records with DecibelsA > 65"""
    return sound_measurements_df[sound_measurements_df["DecibelsA"] > 65]
