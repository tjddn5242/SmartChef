from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import pandas as pd
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('receipe')

def request_query_recipe(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )

    embedded_query=response.data[0].embedding
    results=index.query(embedded_query, top_k=3, include_metadata=True)

    return results['matches']

def request_query_health(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )

    embedded_query=response.data[0].embedding
    results=index.query(embedded_query, top_k=3, include_metadata=True, namespace='health')

    return results['matches']



def gptOutput(user_need, ingredients, disease):

    recipe_request = request_query_recipe(f'{user_need}, Ingredient_Details: {ingredients}')
    recipes = [i['metadata']['text'] for i in recipe_request]

    health_request = request_query_health(f'당뇨병')
    health = [i['metadata']['text'] for i in health_request]

    prompt = f"""
    당신은 한식 요리사 겸 영양사입니다. 당신은 요리 초보자에게 요리 레시피 및 건강정보를 알려줘야합니다.
    요리 초보자는 현재 냉장고에 있는 재료들을 토대로 요리를 하려고 하는데요, 자신의 건강 정보와 현재 땡기는 음식을 기반으로 요리를 만들고 싶어 합니다.

    요리 초보자의 니즈에 맞는 정보를 다음과 같은 순서로 제공해보세요. []는 사용자가 입력하는 정보가 있으며 <>는 당신이 출력해야할 정보입니다.
    1. [재료]와 그 재료로 만들 수 있는 [레시피], 그리고 [현재 땡기는 음식] 정보를 기반으로 <레시피>를 만드세요. [레시피]는 총 3개이므로, <3개의 음식 레시피>를 만드세요.
    2. 각 음식에 필요한 재료 중, [재료]에 <없는 재료>를 찾아서 리스트업하세요.
    3. 요리 초보자의 [몸상태(질병) 정보]와 [질병에 따른 건강정보]를 토대로 <음식별 건강 스코어>를 리스트업하세요.
    4. 마지막으로, 출력한 3개의 음식에 대해 건강 관점에서 <영양사의 한 마디>와 센스있는 <음식명> 출력하세요.

    다음은 입력 정보입니다.
    [재료]
    {ingredients}

    [현재 땡기는 음식]
    {user_need}

    [레시피1]
    {recipes[0]}

    [레시피2]
    {recipes[1]}

    [레시피3]
    {recipes[2]}

    [몸상태(질병) 정보]
    {disease}

    [질병에 따른 건강 정보]
    {health[0]}
    {health[1]}
    {health[2]}

    **중요: 아래 형식에 맞추어 정확히 답변해 주세요. 추가 정보나 텍스트는 추가하지 마세요. 준수할 수 없다면 'N/A'라고 출력해 주세요.**
    <영양사의 한 마디>
    영양사의 한 마디를 출력하세요.

    <음식명1>
    음식명1을 출력하세요.
    - <구비재료1>: [재료]를 출력하세요.
    - <필요재료1>: 없는 재료를 출력하세요.
    - <레시피1>: 레시피를 출력하세요

    <음식명2>
    음식명2을 출력하세요.
    - <구비재료2>: [재료]를 출력하세요.
    - <필요재료2>: 없는 재료를 출력하세요.
    - <레시피2>: 레시피를 출력하세요

    <음식명3>
    음식명3을 출력하세요.
    - <구비재료3>: [재료]를 출력하세요.
    - <필요재료3>: 없는 재료를 출력하세요.
    - <레시피3>: 레시피를 출력하세요
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
        # stream=True
    )

    # for chunk in response:
    #     if chunk.choices[0].delta.content is not None:
    #         return print(chunk.choices[0].delta.content, end="")

    return response.choices[0].message.content


