# Model Limitations

## Key Limitations Based on Error Analysis

1. **Class Imbalance**: The model struggles with minority classes (MN, O, PO) due to limited training examples.
2. **Confusion Pairs**: Common confusions include:
   - F (Functional) confused with SE (Security) and US (Usability)
   - PE (Performance) confused with A (Availability)
3. **Contextual Understanding**: The model lacks deep semantic understanding, often relying on surface-level features.
4. **Overconfidence**: The model shows high confidence in incorrect predictions, especially for majority classes.
5. **Domain-Specific Terms**: The model doesn't fully capture domain-specific terminology that might be important for accurate classification.
