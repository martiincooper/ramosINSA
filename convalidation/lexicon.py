"""Compact multilingual (ES / FR / EN) concept lexicon.

USM syllabi are written in Spanish while INSA Lyon syllabi are written in English
and French. A plain bag-of-words comparison therefore shares almost no tokens
across the two corpora. This lexicon normalises domain vocabulary from the three
languages onto shared canonical concept tokens so the deterministic fallback
agent can still produce meaningful *relative* rankings without an LLM.

It is intentionally domain-focused (industrial / electronic engineering) and not
exhaustive: the LLM reasoning agent remains the recommended engine for final
equivalence decisions. Surface forms are stored accent-stripped and lowercased.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# canonical concept -> surface forms across ES / FR / EN
_CONCEPTS: Dict[str, List[str]] = {
    "production": ["produccion", "production", "fabricacion", "manufacturing", "manufactura"],
    "management": ["gestion", "administracion", "management", "pilotage", "conduite"],
    "operations": ["operaciones", "operations", "operationnel"],
    "research": ["investigacion", "research", "recherche"],
    "optimization": ["optimizacion", "optimisation", "optimization", "optimo", "optimal"],
    "linear_programming": ["lineal", "lineaire", "linear", "programacion", "programmation", "programming"],
    "model": ["modelo", "modelado", "modelamiento", "modelisation", "model", "modeling", "modelling"],
    "inventory": ["inventario", "inventarios", "stock", "stocks", "inventory"],
    "demand": ["demanda", "demande", "demand"],
    "forecast": ["pronostico", "pronosticos", "prevision", "forecast", "forecasting"],
    "quality": ["calidad", "qualite", "quality"],
    "project": ["proyecto", "proyectos", "projet", "project"],
    "supply_chain": ["suministro", "abastecimiento", "approvisionnement", "supply", "chain", "logistica", "logistique", "logistics"],
    "simulation": ["simulacion", "simulation"],
    "statistics": ["estadistica", "estadisticas", "statistique", "statistics", "statistical"],
    "probability": ["probabilidad", "probabilidades", "probabilite", "probability"],
    "cost": ["costo", "costos", "coste", "cout", "couts", "cost"],
    "process": ["proceso", "procesos", "processus", "process"],
    "planning": ["planificacion", "planeacion", "planification", "planning"],
    "control": ["control", "controle", "control"],
    "innovation": ["innovacion", "innovation"],
    "strategy": ["estrategia", "estrategico", "strategie", "strategique", "strategy", "strategic"],
    "marketing": ["marketing", "mercadeo", "mercado", "marche", "market"],
    "finance": ["finanzas", "financiero", "finance", "financiere", "financial"],
    "electronics": ["electronica", "electronique", "electronics", "electronic"],
    "circuit": ["circuito", "circuitos", "circuit", "circuits"],
    "signal": ["senal", "senales", "signal", "signaux", "signals"],
    "design": ["diseno", "conception", "design"],
    "data": ["datos", "donnees", "data"],
    "database": ["base", "bases", "datos", "donnees", "database", "databases"],
    "information": ["informacion", "information"],
    "system": ["sistema", "sistemas", "systeme", "systemes", "system", "systems"],
    "network": ["red", "redes", "reseau", "reseaux", "network", "networks"],
    "automation": ["automatizacion", "automatizado", "automatisation", "automatise", "automation", "automated"],
    "industrial": ["industrial", "industriel", "industria", "industrie", "industry"],
    "engineering": ["ingenieria", "ingenierie", "engineering"],
    "decision": ["decision", "decisiones", "decisions"],
    "risk": ["riesgo", "riesgos", "risque", "risques", "risk"],
    "maintenance": ["mantenimiento", "maintenance"],
    "energy": ["energia", "energie", "energy"],
    "thermodynamics": ["termodinamica", "thermodynamique", "thermodynamics"],
    "programming_code": ["software", "logiciel", "programacion", "code", "informatique", "informatica"],
    "lean": ["lean", "esbelta", "desperdicio", "gaspillage", "waste"],
    "supply_capacity": ["capacidad", "capacite", "capacity"],
    "distribution": ["distribucion", "distribution"],
    "localization": ["localizacion", "localisation", "location"],
    "markov_queue": ["markov", "colas", "files", "queue", "queues", "queueing"],
}

# Build accent-stripped reverse map: surface form -> canonical concept.
_SURFACE_TO_CONCEPT: Dict[str, str] = {}
for _concept, _forms in _CONCEPTS.items():
    for _form in _forms:
        _SURFACE_TO_CONCEPT[strip_accents(_form)] = _concept


def canonical(token: str) -> str:
    """Map a single (lowercased) token to its canonical concept, if known."""
    return _SURFACE_TO_CONCEPT.get(strip_accents(token), token)
