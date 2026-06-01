"""Lightweight tests for the convalidation pipeline.

Runnable with either ``pytest`` or ``python tests/test_pipeline.py``. They cover
the pure parsing / matching logic and do not require the large source PDFs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convalidation import insa_parser, usm_parser, matcher
from convalidation.llm_agent import HeuristicAgent
from convalidation.models import INSACourse, USMCourse
from convalidation.study_plan import build_study_plan


INSA_SHEET = """Domaine Scientifique de la DOUA
Automated production systemsAutomated production systems
IDENTIFICATION
CODE : GI-3-S1-EC-APS
ECTS : 3
HOURS
Cours : 0h
Total : 60h
ASSESMENT METHOD
Oral
TEACHING AIDS
Handout
TEACHING LANGUAGE
French
CONTACT
MME SUBAI Corinne
AIMS
Management of automated production systems and actuators.
CONTENT
Hydraulics, actuators, sensors, control chains, production line design.
BIBLIOGRAPHY
Systemes automatiques
PRE-REQUISITES
None
"""

USM_SHEET = """PROGRAMA DE ASIGNATURA
I. IDENTIFICACION DE LA ASIGNATURA.
Asignatura: Administracion de la Produccion Sigla: ICN-345 Fecha de aprobacion
Creditos UTFSM: 3
Prerrequisitos:
ILN-250
Creditos SCT     : 6  Departamento de
Industrias
Semestre en que se dicta
Impar  Par Ambos
X
Tiempo total de dedicacion a la asignatura: 179 horas cronologicas.
Descripcion de la Asignatura.
El estudiante planifica y controla sistemas de produccion industriales.
Resultados de Aprendizaje que se esperan lograr en esta asignatura.
Planifica la produccion y gestiona inventarios.
Contenidos tematicos.
1. Sistemas de produccion. 2. Pronostico de demanda. 3. Gestion de inventarios.
Metodologia de ensenanza y aprendizaje.
Clases expositivas.
Bibliografia:
Chase, Administracion de la Produccion.
"""


def test_insa_parser_fields():
    course = insa_parser.parse_insa_course(INSA_SHEET)
    assert course is not None
    assert course.code == "GI-3-S1-EC-APS"
    assert course.title == "Automated production systems"
    assert course.ects == 3.0
    assert course.year == "3" and course.semester == "S1"
    assert "Industrial Engineering" in course.department
    assert "actuators" in course.content.lower()
    assert "French" in course.teaching_language


def test_usm_parser_fields():
    course = usm_parser.parse_usm_text(USM_SHEET)
    assert course.code == "ICN-345"
    assert course.title == "Administracion de la Produccion"
    assert course.sct_credits == 6.0
    assert course.utfsm_credits == 3.0
    assert "Industrias" in course.department
    assert "Ambos" in course.semester
    assert "inventarios" in course.contents.lower()
    assert "ILN-250" in course.prerequisites


def test_cross_lingual_similarity_is_positive():
    insa = insa_parser.parse_insa_course(INSA_SHEET)
    usm = usm_parser.parse_usm_text(USM_SHEET)
    agent = HeuristicAgent()
    # Spanish vs French/English production-management courses should share concepts.
    assert agent.score_pair(usm, insa) > 0.0


def test_non_teaching_courses_are_excluded():
    placeholder = INSACourse(code="GI-3-S1-EC-ECH", title="Academic exchange S1", ects=30)
    assert not matcher.is_teachable(placeholder)
    real = insa_parser.parse_insa_course(INSA_SHEET)
    assert matcher.is_teachable(real)


def test_matching_and_status_labels():
    usm = usm_parser.parse_usm_text(USM_SHEET)
    insa_courses = [
        insa_parser.parse_insa_course(INSA_SHEET),
        INSACourse(
            code="GI-3-S1-EC-GIN",
            title="Industrial management",
            ects=4,
            year="3",
            semester="S1",
            department="Industrial Engineering",
            aims="industrial production management and inventory",
            content="production planning, inventory management, demand forecast",
        ),
    ]
    candidates, recs = matcher.match_all([usm], insa_courses, HeuristicAgent())
    assert candidates and recs
    rec = recs[0]
    valid_labels = {
        "Valid",
        "Borderline",
        "Invalid",
        "Valid with combination",
        "Needs additional work",
    }
    assert rec.status in valid_labels
    # The combined ECTS must equal the sum of the chosen INSA courses.
    assert rec.combined_ects == round(
        sum(c.ects for c in insa_courses if c.code in rec.insa_codes), 2
    )


def test_study_plan_groups_by_semester():
    usm = usm_parser.parse_usm_text(USM_SHEET)
    insa_courses = [insa_parser.parse_insa_course(INSA_SHEET)]
    _, recs = matcher.match_all([usm], insa_courses, HeuristicAgent())
    # Force an accepted status so the plan includes the course.
    recs[0].status = "Valid"
    recs[0].insa_codes = ["GI-3-S1-EC-APS"]
    plan = build_study_plan(recs, insa_courses)
    assert plan
    assert plan[0].total_ects == 3.0
    assert any("GI-3-S1-EC-APS" in c for c in plan[0].insa_courses)


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in funcs:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(funcs)} tests passed.")


if __name__ == "__main__":
    _run_all()
