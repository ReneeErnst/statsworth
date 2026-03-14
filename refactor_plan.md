/plan

The RESEARCH.md is excellent and identifies all the technical requirements. Now, please generate a PLAN.md that turns these findings into a step-by-step execution roadmap.

Please organize the plan into the following phases:

Phase 1: Environment Setup (Initialize pyproject.toml, install dependencies, create directory structure).

Phase 2: Foundation Migration (Migrate preprocessing and apply the 'Critical' refactors like the scale_totals fix).

Phase 3: Domain Migration (Migrate factor_analysis, anova, and manova sequentially, including the extraction of shared MANOVA logic).

Phase 4: Visualization & SEM (Migrate plots.py and fit_indices.py).

Phase 5: Test Implementation (Write the new tests identified in the research).

For each step, include the specific verification command (e.g., pytest tests/test_preprocessing.py) that must pass before moving to the next step.

Stop after creating the PLAN.md. Do not start Phase 1 until I approve the roadmap.
