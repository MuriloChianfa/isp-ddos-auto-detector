# Tests

This directory contains test suites for the DDoS Detection System.

## Files

- `__init__.py` - Package initialization
- `test_detection_system.py` - Comprehensive test suite for the detection system
- `run_tests.py` - Test runner script

## Running Tests

### Run all tests:
```bash
cd /root/autoencoders
conda activate nf-ae
python tests/run_tests.py
```

### Run specific test file:
```bash
python -m unittest tests.test_detection_system
```

### Run with verbose output:
```bash
python -m unittest tests.test_detection_system -v
```

## Test Coverage

The test suite covers:

- **Detection System Initialization**: Tests proper system setup
- **Feature Engineering**: Tests feature extraction and preprocessing
- **Model Architecture**: Tests autoencoder model creation and structure
- **Data Preprocessing**: Tests data preparation and scaling
- **Anomaly Detection Interface**: Tests detection method availability
- **Cache Functionality**: Tests caching system
- **Configuration Consistency**: Tests config parameter validation

## Test Structure

- `TestDetectionSystem`: Main system functionality tests
- `TestFeatureEngineering`: Feature engineering specific tests
- `TestModels`: Machine learning model tests

## Adding New Tests

To add new tests:

1. Create a new file with pattern `test_*.py`
2. Import the modules you want to test
3. Create test classes inheriting from `unittest.TestCase`
4. Add test methods with names starting with `test_`
5. Run the tests using the runner script
