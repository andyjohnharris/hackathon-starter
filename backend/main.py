# main.py

import asyncio
import httpx
from dataclasses import dataclass
from asyncpg import Pool
from pydantic_ai import sessions_ta
from pydantic_ai import database_connect
from pydantic_ai import insert_doc_section
from pydantic_ai import AsyncOpenAI
from pydantic_ai import logfire
from pydantic_ai import DOCS_JSON
from pydantic_ai import DB_SCHEMA
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Model and provider configuration
model = OpenAIModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url='https://api.perplexity.ai',
        api_key='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',
    ),
)
agent = Agent(model)

# Database connection pool
@dataclass
class Deps:
    pool: Pool  # Database connection pool

# Build search database
async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    # logfire.instrument_openai(openai)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, openai, pool, section))

async def main():
    print("Hello World!")
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> Paris
    
asyncio.run(main())