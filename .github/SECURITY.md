# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this framework, please report it responsibly by emailing:

**Primary Contact:** murilo.chianfa@uel.br

### What to Include

Please provide the following information:

1. **Description of the vulnerability**: Clear explanation of the security issue
2. **Steps to reproduce**: Detailed steps to reproduce the vulnerability
3. **Potential impact**: How the vulnerability could be exploited
4. **Affected components**: Which parts of the framework are affected
5. **Suggested fix** (optional): If you have ideas on how to fix it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with assessment and expected timeline
- **Resolution**: Depending on severity, typically within 30 days

### Disclosure Policy

- Please allow us reasonable time to investigate and fix the vulnerability before public disclosure
- We will credit you for the discovery (unless you prefer to remain anonymous)
- Once fixed, we will publish a security advisory with details

## Security Considerations for Users

### NetFlow Data Privacy

This framework processes NetFlow data that may contain sensitive information about network traffic:

1. **Data Anonymization**: While the included datasets have been processed to remove direct identifiers, users should:
   - Review datasets for any potentially traceable information
   - Apply additional anonymization if needed for their use case
   - Comply with local data protection regulations (GDPR, LGPD, etc.)

2. **Found Traceable Information**: If you discover any identifiable or traceable information in the provided datasets:
   - **Do not publish or share** the information publicly
   - Contact us immediately at murilo.chianfa@uel.br
   - Provide details about what you found and where
   - We will investigate and take appropriate action

### Secure Usage Practices

1. **Environment Isolation**:
   - Use the provided conda environment (`nf-ae`) to isolate dependencies
   - Keep dependencies updated for security patches
   - Review `environment.yml` before installation

2. **Model Files**:
   - Trained models are saved in `results/` and `cache/` directories
   - These files should be treated as sensitive if trained on proprietary data
   - Use appropriate file permissions in production environments

3. **Network Access**:
   - The framework does not transmit data over the network
   - The framework does not collect telemetry or usage data
   - All processing is local to your machine

4. **Resource Consumption**:
   - Be aware of computational resource usage (CPU, GPU, RAM)
   - Monitor execution in shared or production environments
   - See [README.md](../README.md#security-concerns) for details

### Dependencies

This project uses third-party dependencies managed through Conda and pip:

- Dependencies are pinned to specific versions in `environment.yml`
- Dependabot monitors for security updates in pip packages
- Review security advisories for TensorFlow, PyTorch, and scikit-learn regularly

### Known Limitations

1. **No Authentication**: The framework has no built-in authentication or authorization
2. **Local Execution**: Designed for local execution, not for web deployment without additional security layers
3. **File System Access**: The framework reads/writes local files without sandboxing

## Vulnerability Scope

### In Scope

- Security vulnerabilities in the framework code
- Dependency vulnerabilities that affect functionality
- Data leakage or privacy issues
- Code injection vulnerabilities
- Path traversal issues
- Arbitrary code execution

### Out of Scope

- Issues in third-party dependencies (report to respective maintainers)
- Vulnerabilities requiring physical access to the machine
- Social engineering attacks
- Denial of service through resource exhaustion (expected behavior during training)

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 1.0.1)
- Announced in GitHub Security Advisories
- Documented in release notes with [SECURITY] prefix

## Contact

For security-related questions or concerns:
- **Email**: murilo.chianfa@uel.br
- **Subject Line**: [SECURITY] Brief description

For non-security issues, please use the GitHub issue tracker.

## Acknowledgments

We appreciate the security research community and will acknowledge responsible disclosure in our security advisories.

---

**Note**: This is an academic research project. While we take security seriously, it is provided "as is" without warranties. See [LICENSE](../LICENSE) for details.
