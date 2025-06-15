# import asyncio
# from pydantic_ai import Agent
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.providers.openai import OpenAIProvider

# model = OpenAIModel(
#     'sonar-pro',
#     provider=OpenAIProvider(
#         base_url='https://api.perplexity.ai',
#         api_key='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',
#     ),
# )
# agent = Agent(model)

# result_sync = agent.run_sync('What is the capital of Italy?')
# print(result_sync.output)
#> Rome

# print("Hello World!")
# async def main():
    # print("Hello World!")
    
    # result = await agent.run('What is the capital of France?')
    # print(result.output)
    # #> Paris

    # async with agent.run_stream('What is the capital of the UK?') as response:
    #     print(await response.get_output())
    #     #> London

# asyncio.run(main())

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}
