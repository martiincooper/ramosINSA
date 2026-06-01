# ramosINSA — INSA Lyon ↔ USM syllabus-based convalidation assistant

A pipeline that extracts, structures, and compares **INSA Lyon** course syllabi
against **USM (UTFSM)** course syllabi and produces an Excel file with the best
course *convalidation* proposals.

The pipeline treats **PDF text extraction as the factual source** and an
**LLM-based agent as the reasoning layer**: equivalences are decided from the
actual content of each course sheet, never from titles alone.

## What it does

1. Extracts the text of every page of the bundled INSA courses PDF.
2. Splits it into **individual INSA course sheets** (one PDF + one `.txt` each).
3. Splits every USM syllabus into its own course file.
4. Builds **structured metadata** for every INSA and USM course.
5. Feeds the extracted text to an **agent** that reasons over the content.
6. Ranks INSA candidates, prefers **combinations** when one course is not
   enough, and checks the INSA / USM rules.
7. Produces an **Excel workbook** with 5 sheets of results.

## Academic rules encoded (`convalidation/config.py`)

INSA Lyon selection constraints (max 30 ECTS/semester, at most 2 departments,
~20 ECTS typical, 15 ECTS minimum before internship validation, …) and USM
convalidation requirements (**>= 75 % content equivalence**, **>= 4.5 ECTS per
USM course**, single- or multi-course combinations, thematic coherence) are
encoded as data and shared by the matcher and the LLM prompt.

## The reasoning layer

`convalidation/llm_agent.py` provides two interchangeable agents:

* **`OpenAIAgent`** — used automatically when `OPENAI_API_KEY` is set. It reasons
  over the extracted INSA/USM syllabus texts and returns a structured
  equivalence judgement. Configure the model with `OPENAI_MODEL`
  (default `gpt-4o-mini`).
* **`HeuristicAgent`** — a deterministic, dependency-free fallback based on
  multilingual (ES/FR/EN) concept-normalised bag-of-words similarity. It lets the
  whole pipeline run end-to-end and in CI without any API key. Because USM
  syllabi are Spanish and INSA syllabi are English/French, the fallback gives
  useful *relative rankings* but the **LLM agent is recommended for final
  equivalence decisions**.

## Usage

```bash
pip install -r requirements.txt

# Offline, deterministic run (no API key needed):
python main.py

# LLM-backed reasoning:
export OPENAI_API_KEY=sk-...
python main.py

# Options:
python main.py --insa "INSA COURSES.pdf" --output Results/out.xlsx --quiet
```

## Output

The pipeline creates these folders (regenerated on each run, git-ignored):

* `INSA_Courses/` — one PDF per INSA course sheet.
* `USM_Courses/` — one PDF per USM course.
* `Extracted_Text/INSA/` and `Extracted_Text/USM/` — the extracted text per course.
* `Results/convalidation_proposals.xlsx` — the workbook with:
  1. **USM Courses** — code, name, SCT credits, department, description, key topics.
  2. **INSA Courses** — code, name, ECTS, year, semester, department, key topics.
  3. **Candidate Matches** — USM course, INSA course, similarity score, ECTS, notes.
  4. **Recommended Convalidations** — USM course, recommended INSA course(s),
     combined ECTS, estimated equivalence %, validation status, justification.
  5. **Final Proposed Study Plan** — semester, INSA courses, total ECTS,
     departments involved, target USM convalidations (with rule warnings).

Validation status labels: `Valid`, `Borderline`, `Invalid`,
`Valid with combination`, `Needs additional work`.

## Project layout

```
main.py                     CLI entry point
requirements.txt
convalidation/
  config.py                 paths, source PDF discovery, academic rules
  pdf_utils.py              PDF text extraction & page-range splitting
  models.py                 INSACourse / USMCourse / match data models
  insa_parser.py            split & parse INSA COURSES.pdf course sheets
  usm_parser.py             parse USM "PROGRAMA DE ASIGNATURA" syllabi
  lexicon.py                multilingual concept lexicon (ES/FR/EN)
  llm_agent.py              OpenAI agent + deterministic fallback
  matcher.py                ranking, combinations, validation status
  study_plan.py             preliminary study-plan builder
  excel_writer.py           5-sheet workbook writer
  pipeline.py               end-to-end orchestration
tests/test_pipeline.py      runnable with pytest or `python tests/test_pipeline.py`
```

## Tests

```bash
python tests/test_pipeline.py      # or: pytest
```
