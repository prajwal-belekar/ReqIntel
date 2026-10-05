# RAG Prompt Template for RequirementLens

## Purpose
This prompt template guides the LLM to analyze software requirements with a structured approach:
- Focus on missing specifications
- Consider ML classification as evidence
- Use retrieved guidelines as contextual information
- Avoid hallucinations

## Instructions

### Core Analysis
Analyze the requirement and context below:

1. **Requirement**: The original requirement text
2. **ML Classification**: The ML predicted category and confidence
3. **Guidelines**: Top 3 semantically similar software engineering principles

### Analysis Steps

1. **Identify Missing Specifications**:
   - List all aspects of the requirement that are NOT explicitly stated
   - Include potential missing parameters, constraints, or behaviors
   - Be specific about what is NOT specified

2. **Risk Analysis**:
   - Identify potential risks based on the requirement and context
   - Consider both technical and operational risks
   - Focus on risks that could be mitigated by the ML classification

3. **Recommendations**:
   - Provide actionable recommendations based on the analysis
   - Consider both the ML classification and retrieved guidelines
   - Focus on improving the requirement's completeness

### Hallucination Prevention

**Do NOT assume anything not explicitly stated in the requirement or context.**

**If information is not available:**
- Explicitly state "Not specified in the requirement"
- Explicitly state "Insufficient information to determine this"
- Do NOT invent external facts or assumptions

### Output Format

Provide a clear and structured analysis of the requirement based on the context provided.

Focus on being specific, actionable, and evidence-based in your analysis.

## Context Instructions

The context includes:
- The original requirement text
- The ML classification with confidence
- Top 3 semantically similar software engineering guidelines

**Do NOT use the ML classification as absolute truth.**
**Do NOT assume anything not explicitly stated in the requirement or guidelines.**

## Technical Constraints

1. Do NOT modify the requirement text
2. Do NOT invent new requirements
3. Do NOT assume system capabilities beyond what's specified
4. Do NOT assume domain-specific knowledge not provided in context