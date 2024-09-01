import streamlit as st
from PIL import Image
import os
# from dotenv import load_dotenv
from openai import OpenAI
import base64
import requests
from io import BytesIO
import ast
import time
import json
from llmStructure import *

# .env 파일의 환경 변수들을 불러옵니다.
# load_dotenv()

# OpenAI API Key 설정 (환경 변수 사용)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

def encode_image(image):
    """Encodes the image file to base64 format."""
    if isinstance(image, BytesIO):
        image_data = image.read()  # Read the image data if it's a BytesIO object
    else:
        # If it's a PIL image, convert it to BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_data = buffered.getvalue()
    
    return base64.b64encode(image_data).decode('utf-8')


def recognize_ingredients_from_image(image):
    with st.spinner("Processing image..."):
        time.sleep(2)  # 인코딩 작업 (모의)
        base64_image = encode_image(image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "입력받은 냉장고 속 이미지에서 확실하게 보이는 식재료들만 리스트로 뽑아줘. 이때 식재료와 관련한 이모지를 같이 붙여줘. 불필요한 설명은 제외. format example : ['🥚계란','🎃호박','🍎사과']"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        ingredients_list = response.json()['choices'][0]['message']['content']
        ingredients_list = ast.literal_eval(ingredients_list)
        
    st.success("Done!")
    return ingredients_list

def generate_recipe_response(ingredients, health_condition=None, craving_food=None):
    if not health_condition:
        health_condition = "없음"
    if not craving_food:
        craving_food = "없음"

    prompt = (
        f"""다음 재료들이 있습니다: {', '.join(ingredients)}.
        제 건강 상태는 {health_condition}이고, 현재 {craving_food}을(를) 먹고 싶습니다. 
        이 재료들을 사용하여 만들 수 있는 다양한 요리 레시피를 3개이상 추천해 주세요.

        그리고, 건강상태에 따른 음식 섭취방법이나 주의해야할 재료같은 것도 짧게 한줄로 요약해서 말해줘.
        
        **중요: 아래 형식에 맞추어 정확히 답변해 주세요. 추가 정보나 텍스트는 추가하지 마세요. 준수할 수 없다면 'N/A'라고 출력해 주세요.**

        <output format>
        건강 요약:
        요리 이름:
        조리 시간:
        필요재료:
        추가로 구비해야 하는 재료:
        요리 단계:
        """
    )

    with st.spinner("Generating recipe..."):
        client = OpenAI()
        response = client.chat.completions.create(
        model="gpt-4o-mini",
         messages=[
                {"role": "system", "content": "You are a creative and helpful chef"},
                {"role": "user", "content": prompt}
            ]
        )

    return response.choices[0].message.content

def parse_recipes(gpt_response):
    lines = gpt_response.splitlines()  # 응답을 줄 단위로 나누기
    
    health_summary = None
    recipes = []
    current_recipe = {}
    parsing_steps = False
    
    for line in lines:
        line = line.strip()  # 앞뒤 공백 제거
        
        if line.startswith("건강 요약:"):
            health_summary = line.replace("건강 요약:", "").strip()
        elif line.startswith("요리 이름:"):
            if current_recipe:
                # 마지막으로 파싱된 레시피를 리스트에 추가
                recipes.append(current_recipe)
                current_recipe = {}  # 새로운 레시피 시작을 위해 초기화
            current_recipe["name"] = line.replace("요리 이름:", "").strip()
            parsing_steps = False
        elif line.startswith("조리 시간:"):
            current_recipe["cooking_time"] = line.replace("조리 시간:", "").strip()
        elif line.startswith("필요재료:"):
            current_recipe["all_ingredients"] = line.replace("필요재료:", "").strip()
        elif line.startswith("추가로 구비해야 하는 재료:"):
            current_recipe["additional_ingredients"] = line.replace("추가로 구비해야 하는 재료:", "").strip()
        elif line.startswith("요리 단계:"):
            parsing_steps = True
            current_recipe["steps"] = []
        elif parsing_steps:
            # 요리 단계가 여러 줄에 걸쳐 있을 수 있으므로 리스트로 저장
            current_recipe["steps"].append(line)
    
    # 마지막 레시피 추가
    if current_recipe:
        recipes.append(current_recipe)
    
    # "알 수 없음"으로 표시된 항목들에 대한 기본 처리
    for recipe in recipes:
        recipe["cooking_time"] = recipe.get("cooking_time", "알 수 없음")
        recipe["all_ingredients"] = recipe.get("all_ingredients", "알 수 없음")
        recipe["additional_ingredients"] = recipe.get("additional_ingredients", "없음").replace("N/A", "없음")
        recipe["steps"] = "\n".join(recipe.get("steps", []))

    return health_summary, recipes

# Streamlit 앱 설정
st.set_page_config(page_title="Smart Fridge Recipe Recommender", page_icon="🍽️", layout="wide")

# 제목과 스타일링
st.markdown("<h1 style='text-align: center; color: #FF6347;'>스마트쉐프</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FF4500;'>냉장고에 있는 재료로 최고의 음식을 만들어드립니다</p>", unsafe_allow_html=True)

# 이미지 업로드 기능
st.markdown("### 1. 냉장고 사진을 업로드 해주세요")
# img_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
img_file = 'uploaded_image.jpg' # 디버깅용 ===========================================================================

# 사진이 삭제되었는지 확인 (img_file이 None인 경우)
if img_file is None:
    if 'ingredients' in st.session_state:
        st.session_state.ingredients = []  # 재료 리스트 초기화

if img_file is not None:
    img = Image.open(img_file)

    # RGBA 이미지를 RGB로 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    # 이미 인식된 재료가 없는 경우에만 이미지 인식 수행
    if 'ingredients' not in st.session_state or not st.session_state.ingredients:
        # detected_ingredients = recognize_ingredients_from_image(img)
        detected_ingredients = ["🥔감자", "🥚달걀", "🫑파프리카", "🥒오이", "🌶️고추", "🥕당근"] # 디버깅용 ===========================================================================
        st.write("Recognized Ingredients:")
        st.write(detected_ingredients) # 디버깅용 ===========================================================================
        st.session_state.ingredients = list(set(detected_ingredients))
    
        # Detected Ingredients Display (5 items per row)
        st.markdown("### 2. 인식된 재료들을 확인해보세요.")


    # Ensure the session state is set up correctly
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = ingredients

    with st.expander("각 재료 옆의 x버튼을 눌러 잘못 인식된 재료들을 삭제 할 수 있습니다.", expanded=True):
        if st.session_state.ingredients:
            remove_indices = []
            rows = len(st.session_state.ingredients) // 5 + 1
            for i in range(rows):
                cols = st.columns(5)
                for j in range(5):
                    idx = i * 5 + j
                    if idx < len(st.session_state.ingredients):
                        ingredient = st.session_state.ingredients[idx]
                        with cols[j]:
                            # 하나의 컨테이너에 ingredient와 X 버튼을 함께 담기
                            container = st.container(border=True)
                            with container:
                                col1, col2 = st.columns([4, 1])
                                col1.markdown(f"<p style='font-size:16px;'>{ingredient}</p>", unsafe_allow_html=True)
                                if col2.button('X', key=f"remove_{ingredient}_{idx}"):
                                    remove_indices.append(idx)


            if remove_indices:
                # 인덱스를 역순으로 정렬 후 pop으로 삭제하여 인덱스 오류 방지
                for idx in sorted(remove_indices, reverse=True):
                    st.session_state.ingredients.pop(idx)
                
                # 페이지를 다시 로드하여 UI 업데이트
                # st.query_params(updated="true")
        else:
            st.markdown("<p style='font-size:16px;'>No ingredients detected yet. Please upload an image.</p>", unsafe_allow_html=True)

    # 재료 추가 기능
    st.markdown("### 3. 인식하지 못한 재료들을 입력해서 추가해보세요.")
    new_ingredients = st.text_input("여러 재료를 입력할 때는 콤마(,)를 이용해주세요", placeholder="EX.계란, 숙주, 소세지")
    if st.button("재료 추가하기"):
        if new_ingredients:
            new_ingredients_list = [ingredient.strip() for ingredient in new_ingredients.split(',')]
            st.session_state.ingredients.extend(new_ingredients_list)
            st.session_state.ingredients = list(set(st.session_state.ingredients))
            st.rerun()  # UI 업데이트

    # 사용자 건강 상태와 땡기는 음식 입력 받기
    st.markdown("### 4. 가지고 있는 질병과 현재 땡기는 음식을 말씀해주세요.")
    health_condition = st.text_input("가지고 있는 질병이 있다면 입력해주세요 (ex. 당뇨병, 야맹증, 고혈압 등)", placeholder="없다면 입력하지 않으셔도 됩니다")
    craving_food = st.text_input("지금 땡기는 음식이 있다면 입력해주세요", placeholder="없다면 입력하지 않으셔도 됩니다")

    # Analyze 버튼
    if st.button("음식을 추천해줘", help="Click to find recipes based on your ingredients and preferences"):
        if st.session_state.ingredients:

            gpt_response = json.loads(gptOutput(craving_food, st.session_state.ingredients, health_condition)[0])
            health_summary = gpt_response['chefTip']
            recipes = gpt_response['recipes']

            # print(gpt_response)
            # print(health_summary)
            # print(recipes)

            # 건강 요약 부분을 별도로 출력
            if health_summary:
                st.markdown("### 건강 요약")
                st.markdown(f"**{health_summary}**")
                st.markdown("---")  # 구분선을 추가하여 건강 요약과 레시피를 구분

            st.markdown("### 추천 레시피")

            cols = st.columns(3)  # 3개의 열로 카드 형식의 레이아웃 생성

            for i, recipe in enumerate(recipes.values()):
                with cols[i % 3]:
                    st.markdown(f"<h3 style='color: #FF4500;'>{recipe['name']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"조리시간: {recipe['cooking_time']}")
                    st.markdown(f"필요재료: {recipe['all_ingredients']}")
                    st.markdown(f"추가구비재료: {recipe['additional_ingredients']}")

                    # Expander 사용하여 준비 단계 표시
                    with st.expander("조리방법보기"):
                        st.markdown("#### 조리 방법")
                        # 조리 단계에서 줄바꿈 적용하여 표시
                        steps = recipe['steps'].split('\n')
                        for step in steps:
                            st.markdown(f"{step.strip()}")

else:
    st.warning("먼저 사진을 업로드 해주세요")

st.write(type(recipes))
st.write(recipes)

user_need = craving_food
ingredients = st.session_state.ingredients
disease = health_condition

st.markdown(f'''
### 변수
- user_need: {user_need}
- ingredients: {ingredients}
- disease: {disease}
---
### 프롬프트
{gptOutput(user_need, ingredients, disease)[1]}
---
### 모델output
{gptOutput(user_need, ingredients, disease)[0]}
''')
