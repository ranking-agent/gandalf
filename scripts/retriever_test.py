import asyncio
import json
import time
from datetime import datetime

import httpx

# retriever_url = "https://dev.retriever.biothings.io/query"
retriever_url = "http://localhost:8080/query"


def generate_query(curie: str) -> dict:
    """Given a curie, return a TRAPI message."""
    parameters = {
        # "timeout": 210,
        "tiers": [0],
    }
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "SN": {"categories": ["biolink:ChemicalEntity"]},
                    "ON": {
                        "ids": ["MONDO:0007186"],
                        "categories": ["biolink:DiseaseOrPhenotypicFeature"],
                    },
                    "e": {"categories": ["biolink:ChemicalEntity"]},
                    "i": {"categories": ["biolink:BiologicalEntity"]},
                },
                "edges": {
                    "edge_0": {
                        "subject": "e",
                        "object": "ON",
                        "predicates": ["biolink:treats_or_applied_or_studied_to_treat"],
                    },
                    "edge_1": {
                        "subject": "i",
                        "object": "SN",
                        "predicates": ["biolink:affects"],
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "abundance",
                                    },
                                    {
                                        "qualifier_type_id": "biolink:object_direction_qualifier",
                                        "qualifier_value": "increased",
                                    },
                                ]
                            }
                        ],
                    },
                    "edge_2": {
                        "subject": "i",
                        "object": "e",
                        "predicates": ["biolink:affects"],
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity_or_abundance",
                                    },
                                    {
                                        "qualifier_type_id": "biolink:object_direction_qualifier",
                                        "qualifier_value": "increased",
                                    },
                                ]
                            }
                        ],
                    },
                },
            }
        },
        "parameters": parameters,
        "log_level": "DEBUG",
    }


async def single_lookup(curie: str):
    """Run a single query lookup synchronously."""
    query = generate_query(curie)
    start_time = datetime.now()
    try:
        async with httpx.AsyncClient(timeout=600000) as client:
            response = await client.post(
                retriever_url,
                json=query,
            )
            response.raise_for_status()
            response = response.json()
            num_results = len((response.get("message") or {}).get("results") or [])
    except Exception as e:
        num_results = 0
        response = {
            "Error": str(e),
        }

    stop_time = datetime.now()
    print(
        f"{curie} took {stop_time - start_time} seconds and gave {num_results} results"
    )
    with open(f"retriever/{curie}_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f)


async def main():
    """Run the given query and time it."""
    start = time.time()
    num_queries = 1
    queries = [single_lookup("MONDO:0007186") for _ in range(num_queries)]
    await asyncio.gather(*queries)
    print(f"All queries took {time.time() - start} seconds")


if __name__ == "__main__":
    asyncio.run(main())

# CHEBI:45783 Imatinib
