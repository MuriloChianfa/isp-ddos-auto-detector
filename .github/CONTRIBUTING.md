# Contributing to ISP DDoS Auto-Detector

Thank you for your interest in contributing to this research framework! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.12.2
- Conda (Miniconda or Anaconda)
- Git and Git LFS
- Linux environment (recommended)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/MuriloChianfa/isp-ddos-auto-detector.git
   cd isp-ddos-auto-detector
   ```

2. Install Git LFS and pull datasets:
   ```bash
   git lfs install
   git lfs pull
   ```

3. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate nf-ae
   ```

4. Verify the installation:
   ```bash
   python main.py --help
   ```

## How to Contribute

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/BUG_REPORT_TEMPLATE.md) and include:
- Environment details (Python version, OS, GPU)
- Dataset and configuration used
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and logs

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/FEATURE_REQUEST_TEMPLATE.md) and describe:
- Research motivation for the feature
- How it improves DDoS detection
- Affected algorithms or datasets
- Potential implementation approach

### Submitting Changes

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Test your changes**:
   ```bash
   conda activate nf-ae
   python main.py --dataset itp-downstream-http-flood --model autoencoder --time-window 1seconds
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add: description of your changes"
   ```

5. **Push and create a Pull Request** using the appropriate template

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use descriptive variable names
- Add docstrings for functions and classes

### Example:
```python
def calculate_entropy(data: np.ndarray, base: int = 2) -> float:
    """
    Calculate Shannon entropy of the input data.
    
    Args:
        data: Input array of values
        base: Logarithm base (default: 2 for bits)
        
    Returns:
        Shannon entropy value
    """
    # Implementation here
    pass
```

### Framework Integration

- Place new models in `framework/models/`
- Add visualization functions to `framework/visualization/`
- Update `config.py` for new datasets or hyperparameters
- Maintain compatibility with existing pipeline in `framework/pipeline.py`

### Testing

- Test with all three datasets when applicable
- Verify results match expected performance metrics
- Test with different temporal windows (1s, 10s, 60s, 300s)
- Ensure GPU and CPU execution both work

## Pull Request Process

1. **Use the appropriate PR template**:
   - [Default Template](.github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md)
   - [Bug Fix Template](.github/PULL_REQUEST_TEMPLATE/BUG_FIX_TEMPLATE.md)
   - [Feature Template](.github/PULL_REQUEST_TEMPLATE/FEATURE_TEMPLATE.md)

2. **Ensure all checklist items are completed**:
   - Code runs in `nf-ae` conda environment
   - Changes tested with relevant datasets
   - Documentation updated if needed
   - No breaking changes to existing functionality

3. **Wait for review** from maintainers

4. **Address review feedback** promptly

## Dataset Contributions

If you have NetFlow data from confirmed DDoS attacks:

1. **Anonymize the data** to remove sensitive information
2. Contact maintainers at murilo.chianfa@uel.br to discuss inclusion
3. Follow the existing dataset structure in `datasets/`
4. Provide attack metadata (type, duration, characteristics)
5. Include both raw NetFlow and derived features if possible

## Research and Academic Use

This is an academic research project. When contributing:

- Cite relevant academic papers for new algorithms
- Document research methodology for new features
- Maintain reproducibility with pinned dependencies
- Consider computational efficiency for large-scale datasets

## Maintainers

- **Murilo A. Chianfa** (murilo.chianfa@uel.br) - State University of Londrina (UEL)
- **Rodrigo S. Miani** (miani@ufu.br) - Federal University of Uberlândia (UFU)
- **Bruno B. Zarpelão** (brunozarpelao@uel.br) - State University of Londrina (UEL)

## License

- **Code contributions**: Licensed under MIT License
- **Dataset contributions**: Licensed under Creative Commons Attribution 4.0 (CC BY 4.0)

By contributing, you agree to license your contributions under these terms.

## Questions?

Feel free to open an issue for questions or contact the maintainers directly.

Thank you for helping improve DDoS detection research! 🛡️
