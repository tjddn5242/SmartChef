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

    health_request = request_query_health(disease)
    health = [i['metadata']['text'] for i in health_request]

    print(health)

    prompt = f"""
    당신은 한식 요리사 겸 영양사입니다. 당신은 요리 초보자에게 요리 레시피 및 건강정보를 알려줘야합니다.
    요리 초보자는 현재 냉장고에 있는 재료들을 토대로 요리를 하려고 하는데요, 자신의 건강 정보와 현재 땡기는 음식을 기반으로 요리를 만들고 싶어 합니다.

    요리 초보자의 니즈에 맞는 정보를 다음과 같은 순서로 제공해보세요. []는 사용자가 입력하는 정보입니다.
    1.  [재료]와 그 재료로 만들 수 있는 [레시피], 그리고 [현재 땡기는 음식], 그리고 [질병에 따른 건강정보] 를 기반으로 '레시피(recipes)'를 만드세요.  레시피는 총 3개를 만들어야 하며, 각 레시피에는 '요리 제목(name, english_name)', '추가 구비 재료(additional_ingredients)', '전체 필요 재료(all_ingredients)', '요리 시간(cooking_time)', '요리 단계(step)', '건강 점수(health_score)'이 필요합니다.
    2. [레시피]를 참고해서 '요리 제목', '추가 구비 재료', '전체 필요 재료', '요리 시간', '요리 단계'를 생성하세요. '요리 제목'은 요리의 제목을 의미하고, '추가 구비 재료'는 [재료]에는 없지만 요리에 필요한 재료를 의미하고, '전체 필요 재료'는 요리에 필요한 모든 재료를 의미하고, '요리 시간'은 요리에 필요한 시간을 의미합니다. '요리 단계'는 최대한 자세하게(줄넘김으로 출력) 초보자도 이해하기 쉽게 설명하세요.  '건강 점수'는 [질병에 따른 건강정보]를 토대로 계산해주세요. '전체 필요 재료'에 [질병에 따른 건강정보]에 있는 주의식품이 많으면 점수가 깎이고, 권장식품이 많으면 점수가 올라갑니다. 그리고, 예시로 4점은 ★★★★☆ 같이 표현합니다.
    3. 요리 초보자의 [몸상태(질병) 정보]와 [질병에 따른 건강정보]를 토대로 '쉐프의 한 마디(chefTip)'를 출력하세요. '쉐프의 한 마디'는 전체 요리에 대해 건강 관점에서 음식에 대한 설명을 해줘야합니다. '쉐프의 한 마디'는 최대한 자세하고 길게 설명해주고, 주의식품에 대해 강조해서 설명해주세요. 그리고 가독성이 있도록 적절하게 줄바꿈을 해주세요.

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

    **중요: 아래와 같이 json형태로 출력하세요. 이외에는 그 어떤 말도 출력하지 마세요.**
    {{"chefTip":"","recipes":{{"first":{{"english_name":"", "name":"","additional_ingredients":"","all_ingredients":"", "steps":"","cooking_time":"", "health_score":""}} ,"second":{{"english_name":"", "name":"","additional_ingredients":"","all_ingredients":"","steps":"","cooking_time":"", "health_score":""}} ,"third":{{"english_name":"", "name":"","additional_ingredients":"","all_ingredients":"","steps":"","cooking_time":"", "health_score":""}}}}}}
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

    return (response.choices[0].message.content, prompt)