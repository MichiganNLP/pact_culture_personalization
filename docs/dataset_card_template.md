---
license: cc-by-4.0
task_categories:
- text-classification
- question-answering
language:
- en
pretty_name: PACT Culture-Personalization Benchmark
---

# PACT: Personal-Preference and Cultural-Norm Trade-off

PACT contains culturally grounded social scenarios where a model chooses between following a cultural expectation and allowing a personal preference.

## Data Fields

- `scenario_id`: stable instance identifier
- `scenario`: social situation
- `actor_country`, `receiver_country`: actor and receiver country context
- `actor_age`, `actor_gender`, `receiver_age`, `receiver_gender`: demographic attributes
- `cultural_expectation`: local cultural norm
- `personal_preference`: preference in tension with the norm
- `choice_a`, `choice_b`: candidate actions when applicable
- `metadata_source`: source family used to construct the item

## Labels / Decisions

Model evaluations use:

- `FOLLOW_CULTURE`
- `ALLOW_PREFERENCE`

Human-study alignment files use A/B choices mapped to culture-following or preference-allowing actions.

## Intended Use

This dataset is intended for research on cultural alignment, personalization, pluralistic evaluation, and uncertainty/disagreement in social judgment.

## Limitations

PACT should not be used as a prescriptive source of cultural rules. Cultural expectations are context-dependent and internally diverse; the benchmark is designed to measure model behavior under trade-offs, not to define correct behavior for real people.
