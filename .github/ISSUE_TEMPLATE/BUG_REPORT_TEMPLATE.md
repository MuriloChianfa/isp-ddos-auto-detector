---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

<!-- A clear and concise description of the bug -->

## Environment

**Python Version:**
```bash
python --version
```

**Operating System:**
- [ ] Linux (specify distribution): 
- [ ] macOS (specify version): 
- [ ] Windows (specify version): 

**GPU:**
- [ ] NVIDIA GPU (specify model): 
- [ ] CPU only

**CUDA Version (if applicable):**
```bash
nvidia-smi
```

**Conda Environment:**
```bash
conda activate nf-ae
conda list | grep -E "tensorflow|scikit-learn|pandas"
```

## Dataset

**Which dataset were you using?**
- [ ] itp-downstream-http-flood
- [ ] itp-multivector-udp-100gbps-peak
- [ ] itp-synack-customer-outage
- [ ] Custom dataset

## Configuration

**Temporal Window:**
- [ ] 1 second
- [ ] 10 seconds
- [ ] 60 seconds
- [ ] 300 seconds

**PCC Threshold:**
- [ ] 0.50
- [ ] 0.70
- [ ] 0.90
- [ ] Other: 

**Detection Algorithm:**
- [ ] Autoencoder
- [ ] Isolation Forest
- [ ] Local Outlier Factor
- [ ] One-Class SVM

## Steps to Reproduce

1. 
2. 
3. 
4. 

**Command executed:**
```bash
conda activate nf-ae
# Paste the exact command that triggers the bug

```

## Expected Behavior

<!-- What you expected to happen -->

## Actual Behavior

<!-- What actually happened -->

## Error Message

```
<!-- Paste the full error message and stack trace here -->

```

## Logs

<!-- If applicable, include relevant log files or output -->

```
<!-- Paste logs here -->

```

## Screenshots

<!-- If applicable, add screenshots to help explain the problem -->

## Reproducibility

**How often does this bug occur?**
- [ ] Always
- [ ] Sometimes
- [ ] Only once

**Can you reproduce it consistently?**
- [ ] Yes
- [ ] No

## Workaround

<!-- If you found a temporary workaround, please describe it -->

## Additional Context

<!-- Add any other context about the problem -->

## Checklist

- [ ] I have activated the `nf-ae` conda environment
- [ ] I have pulled the latest changes from the repository
- [ ] I have checked existing issues to avoid duplicates
- [ ] I have included all relevant error messages
- [ ] I have provided steps to reproduce the bug
