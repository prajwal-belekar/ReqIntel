# PROMISE Dataset Class Labels

This document provides the authoritative mapping of class labels used in the RequirementLens project, based on the PROMISE dataset for Non-Functional Requirements classification.

## Class Definitions

| Label | Meaning | Number of Samples | Description |
|-------|---------|-------------------|-------------|
| A | Availability | 31 | System availability and uptime requirements |
| F | Functional | 444 | Functional requirements specifying system behavior |
| FT | Fault Tolerance | 18 | System fault tolerance and error handling requirements |
| L | Legal | 15 | Legal and compliance requirements |
| LF | Look and Feel | 49 | User interface appearance and design requirements |
| MN | Maintainability | 24 | Code maintainability and software evolution requirements |
| O | Operational | 77 | System operational and administrative requirements |
| PE | Performance Efficiency | 67 | Performance and efficiency requirements |
| PO | Portability | 12 | System portability across platforms |
| SC | Scalability | 22 | System scalability requirements |
| SE | Security | 125 | Security and access control requirements |
| US | Usability | 85 | User experience and usability requirements |

## Dataset Statistics

- **Total Requirements**: 969
- **Unique Classes**: 12
- **Unique Projects**: 47
- **Class Imbalance**: Significant (F: 444 samples, PO: 12 samples)

## Source

The class labels are derived from the PROMISE dataset for Non-Functional Requirements classification. The dataset contains requirements extracted from various software projects with manual annotations for requirement categories.

> **Note**: These class definitions are used consistently across all RequirementLens experiments. The ML baseline model (Experiment 2) uses these exact labels with class_weight="balanced" to address class imbalance.
