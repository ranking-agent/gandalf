import asyncio
import json
import os
import time

import httpx

with open("./benchmark_queries.json", "r", encoding="utf-8") as f:
    messages = json.load(f)
# with open("./pathfinder_queries.json", "r", encoding="utf-8") as f:
#     messages = json.load(f)

urls = {
    "retriever_tier_0": "https://dev.retriever.biothings.io/query",
    "retriever_tier_1": "https://dev.retriever.biothings.io/query",
    "gandalf": "https://automat.renci.org/translatorkg-gandalf/query",
}

do_concurrent = False


async def do_lookup(target, url, message, indx):
    t0 = time.perf_counter()
    if "log_level" in message:
        del message["log_level"]
        message["log_level"] = "INFO"
    if target == "retriever_tier_1":
        message["parameters"]["tiers"] = [1]
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                url,
                json=message,
            )
            with open(f"{target}/{indx}_benchmark_response.json", "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
            response.raise_for_status()
            response = response.json()
            num_results = len((response.get("message") or {}).get("results") or [])
            print(f"Returned {num_results} results")
    except httpx.ReadTimeout:
        print("Timed out")
        num_results = 0
    except httpx.HTTPError:
        print("Got bad response:", response.content)
        num_results = 0
    t1 = time.perf_counter()
    if not do_concurrent:
        print(f"Query took {t1 - t0} seconds")
    return t1 - t0, num_results


async def main():
    """Run the given query and time it."""
    for target, url in urls.items():
        os.makedirs(target, exist_ok=True)
        total_time = time.perf_counter()
        all_results = 0
        if do_concurrent:
            queries = []
            for indx, message in enumerate(messages):
                queries.append(do_lookup(target, url, message, indx))
            responses = await asyncio.gather(*queries)

            for response in responses:
                all_results += response[1]
        else:
            for indx, message in enumerate(messages):
                _, num_results = await do_lookup(target, url, message, indx)
                all_results += num_results
        print(f"{target} Total time:", time.perf_counter() - total_time)
        print(f"{target} Total results:", all_results)


if __name__ == "__main__":
    asyncio.run(main())
