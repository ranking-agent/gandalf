"""Locust stress tests for the GANDALF TRAPI server.

Usage:
    # Web UI (default: http://localhost:8089)
    locust -f stress_tests/locustfile.py --host http://localhost:6429

    # Headless: 50 users, spawn rate 5/s, run for 60s
    locust -f stress_tests/locustfile.py --host http://localhost:6429 \
        --headless -u 50 -r 5 -t 60s
"""

import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Sample data drawn from the GANDALF test fixtures / TRAPI examples
# ---------------------------------------------------------------------------

SAMPLE_CURIES = [
    "CHEBI:6801",  # Metformin (SmallMolecule/Drug)
    "NCBIGene:5468",  # PPARG (Gene)
    "NCBIGene:3643",  # INSR (Gene)
    "MONDO:0005148",  # Type 2 Diabetes (Disease)
    "HP:0001943",  # Hypoglycemia (PhenotypicFeature)
    "CHEBI:17234",  # Glucose (SmallMolecule)
    "GO:0006006",  # Glucose metabolic process (BiologicalProcess)
    "NCBIGene:2645",  # GCK (Gene)
    "NCBIGene:7124",  # TNF (Gene)
    "MONDO:0005015",  # Diabetes Mellitus (Disease)
    "MONDO:0004995",  # Cardiovascular Disease (Disease)
]

GENE_CURIES = [c for c in SAMPLE_CURIES if c.startswith("NCBIGene:")]
DISEASE_CURIES = [c for c in SAMPLE_CURIES if c.startswith("MONDO:")]
DRUG_CURIES = [c for c in SAMPLE_CURIES if c.startswith("CHEBI:")]

PREDICATES = [
    "biolink:affects",
    "biolink:treats",
    "biolink:interacts_with",
    "biolink:related_to",
    "biolink:gene_associated_with_condition",
    "biolink:has_phenotype",
]

CATEGORIES = [
    "biolink:Gene",
    "biolink:Disease",
    "biolink:SmallMolecule",
    "biolink:PhenotypicFeature",
    "biolink:BiologicalProcess",
]


def _onehop_query(subject_ids, object_categories, predicates):
    """Build a one-hop TRAPI query body."""
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": subject_ids},
                    "n1": {"categories": object_categories},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": predicates,
                    }
                },
            }
        }
    }


def _twohop_query(subject_ids, mid_categories, end_categories, pred1, pred2):
    """Build a two-hop TRAPI query body."""
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": subject_ids},
                    "n1": {"categories": mid_categories},
                    "n2": {"categories": end_categories},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": pred1,
                    },
                    "e1": {
                        "subject": "n1",
                        "object": "n2",
                        "predicates": pred2,
                    },
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# Locust user
# ---------------------------------------------------------------------------


class GandalfUser(HttpUser):
    """Simulates a client hitting the GANDALF TRAPI endpoints."""

    wait_time = between(0.5, 2.0)

    # -- lightweight / read-only endpoints ----------------------------------

    @task(0)
    def metadata(self):
        self.client.get("/metadata")

    @task(0)
    def meta_knowledge_graph(self):
        self.client.get("/meta_knowledge_graph")

    # -- TRAPI queries (the heavy hitters) ----------------------------------

    @task(5)
    def query_onehop(self):
        subject = random.choice(DRUG_CURIES + GENE_CURIES)
        obj_cat = random.choice(["biolink:Gene", "biolink:Disease"])
        pred = random.choice(PREDICATES[:3])  # affects, treats, interacts_with
        body = _onehop_query([subject], [obj_cat], [pred])
        self.client.post("/query", json=body, name="/query [1-hop]")

    @task(0)
    def query_twohop(self):
        subject = random.choice(DRUG_CURIES)
        body = _twohop_query(
            subject_ids=[subject],
            mid_categories=["biolink:Gene"],
            end_categories=["biolink:Disease"],
            pred1=["biolink:affects"],
            pred2=["biolink:gene_associated_with_condition"],
        )
        self.client.post("/query", json=body, name="/query [2-hop]")

    @task(0)
    def query_onehop_with_qualifiers(self):
        subject = random.choice(DRUG_CURIES)
        body = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": [subject]},
                        "n1": {"categories": ["biolink:Gene"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {
                                            "qualifier_type_id": "biolink:object_aspect_qualifier",
                                            "qualifier_value": "activity",
                                        }
                                    ]
                                }
                            ],
                        }
                    },
                }
            }
        }
        self.client.post("/query", json=body, name="/query [1-hop+qualifiers]")
