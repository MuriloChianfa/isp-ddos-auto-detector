## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues

<!-- Link related issues using #issue_number -->

Fixes #
Relates to #

## Changes Made

<!-- List the specific changes made in this PR -->

- 
- 
- 

## Affected Components

<!-- Mark all that apply -->

- [ ] Detection algorithms (Autoencoder, Isolation Forest, LOF, One-Class SVM)
- [ ] Feature engineering (`framework/features.py`)
- [ ] Evaluation metrics (`framework/evaluation.py`)
- [ ] Visualization (`framework/visualization/`)
- [ ] Pipeline (`framework/pipeline.py`)
- [ ] Configuration (`config.py`)
- [ ] Datasets
- [ ] Documentation

## Testing

<!-- Describe the tests you ran to verify your changes -->

### Environment
- Python version: 
- OS: 
- GPU: Yes/No

### Test Commands
```bash
conda activate nf-ae
# Add commands used for testing

```

### Datasets Tested
- [ ] itp-downstream-http-flood
- [ ] itp-multivector-udp-100gbps-peak
- [ ] itp-synack-customer-outage

### Temporal Windows Tested
- [ ] 1 second
- [ ] 10 seconds
- [ ] 60 seconds
- [ ] 300 seconds

## Checklist

<!-- Mark completed items with an "x" -->

- [ ] Code runs successfully in `nf-ae` conda environment
- [ ] Changes tested with at least one dataset
- [ ] Code follows project style guidelines (PEP 8, 4 spaces, max 120 chars)
- [ ] Docstrings added/updated for new functions/classes
- [ ] No breaking changes to existing functionality
- [ ] Results are reproducible
- [ ] Documentation updated (if needed)
- [ ] No sensitive data or credentials included
- [ ] Git LFS used for large files (if applicable)

## Performance Impact

<!-- Describe any performance implications -->

- Execution time: 
- Memory usage: 
- GPU utilization: 

## Screenshots/Results

<!-- If applicable, add screenshots or result tables -->

## Additional Notes

<!-- Any additional information reviewers should know -->

## Reviewer Checklist

<!-- For maintainers -->

- [ ] Code quality and style verified
- [ ] Tests pass successfully
- [ ] Documentation is clear and complete
- [ ] No security concerns
- [ ] Changes align with project goals
