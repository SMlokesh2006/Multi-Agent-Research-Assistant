
import asyncio
import os
from dotenv import load_dotenv
from tavily import AsyncTavilyClient

load_dotenv()

async def test_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    print(f"Testing Tavily with key: {api_key[:10]}...")
    client = AsyncTavilyClient(api_key=api_key)
    try:
        response = await client.search(query="latest news on AI", max_results=3)
        results = response.get("results", [])
        print(f"Success! Found {len(results)} results.")
        for r in results:
            print(f"- {r.get('title')}")
    except Exception as e:
        print(f"Tavily Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tavily())
