import streamlit as st
from PIL import Image
import openai
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from dotenv import load_dotenv

# .env 파일의 환경 변수들을 불러옵니다.
load_dotenv()

# OpenAI API Key 설정 (환경 변수 사용)
openai.api_key = os.getenv('OPENAI_API_KEY')

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 확장된 재료 리스트
ingredient_list = [
    "lettuce", "tomato", "cucumber", "olive oil", "banana", "strawberry", "yogurt", "honey",
    "cheese", "bread", "egg", "chicken", "beef", "pork", "fish", "garlic", "onion", "carrot",
    "potato", "bell pepper", "spinach", "mushroom", "avocado", "rice", "pasta", "milk", "butter",
    "flour", "sugar", "salt", "pepper", "chocolate", "bacon", "sausage", "apple", "orange", "grapes",
    "peanut butter", "almond", "walnut", "blueberry", "raspberry", "blackberry", "cabbage", "zucchini"
]

def recognize_ingredients_from_image(image):
    try:
        inputs = processor(text=ingredient_list, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        recognized_ingredients = []
        threshold = 0.01  # 예제 기준, 재료 인식 확률이 1% 이상일 경우
        for i, prob in enumerate(probs[0]):
            if prob > threshold:
                recognized_ingredients.append(ingredient_list[i])

        return recognized_ingredients
    except Exception as e:
        st.error(f"Error in ingredient recognition: {e}")
        return []

def generate_recipe_response(ingredients, health_condition=None, craving_food=None):
    if not health_condition:
        health_condition = "None"
    if not craving_food:
        craving_food = "None"

    prompt = (
        f"""I have the following ingredients: {', '.join(ingredients)}.
        My health condition is {health_condition}, and I am currently craving {craving_food}. 
        Please suggest many recipes that I can make with these ingredients, considering my health and craving.
        
        **Important: Follow the exact format below. Do not add any extra information or text. If you can't comply, just output 'N/A'.**

        <Output Format>
        - Recipe: 
        - Cooking time:
        - Additional ingredients:
        - Steps:
        """
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative and helpful chef."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def parse_recipes(gpt_response):
    # GPT의 응답을 파싱하여 레시피 정보를 추출합니다.
    recipes = []
    recipe_blocks = gpt_response.split("\n\n")  # 공백 두 줄로 레시피 블록을 나눕니다.

    for block in recipe_blocks:
        lines = block.strip().split("\n")
        name = None
        cooking_time = "Unknown"
        additional_ingredients = "None"
        steps_start = 0

        for i, line in enumerate(lines):
            if "Recipe:" in line:
                name = line.replace("Recipe:", "").strip()
            if "Cooking time:" in line:
                cooking_time = line.replace("Cooking time:", "").strip()
            if "Additional ingredients:" in line:
                additional_ingredients = line.replace("Additional ingredients:", "").strip()
            if "Steps:" in line:
                steps_start = i + 1
                break

        if not name:
            name = lines[0].strip()

        steps = "\n".join(lines[steps_start:])
        recipes.append({
            "name": name,
            "cooking_time": cooking_time,
            "additional_ingredients": additional_ingredients,
            "steps": steps
        })

    return recipes

# Streamlit 앱 설정
st.title("Smart Fridge Recipe Recommender")

# 초기 상태 설정
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = []

# 이미지 업로드 기능
# img_file = st.file_uploader("Upload a picture of your fridge", type=["jpg", "jpeg", "png"])
img_file = 'uploaded_image.jpg'

if img_file is not None:
    img = Image.open(img_file)

    # RGBA 이미지를 RGB로 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 이미지에서 재료 인식
    detected_ingredients = recognize_ingredients_from_image(img)
    st.session_state.ingredients.extend(detected_ingredients)
    st.session_state.ingredients = list(set(st.session_state.ingredients))  # 중복 제거

# 동적으로 재료 관리
with st.expander("Detected ingredients", expanded=True):
    for i, ingredient in enumerate(st.session_state.ingredients):
        col1, col2 = st.columns([10, 1])
        with col1:
            st.write(ingredient)
        with col2:
            if st.button('X', key=f"remove_{ingredient}_{i}"):
                st.session_state.ingredients.pop(i)
                st.rerun()  # UI 업데이트

# 재료 추가 기능
with st.expander("Add new ingredients"):
    new_ingredients = st.text_input("Enter new ingredients (comma-separated)")
    if st.button("Add Ingredients"):
        if new_ingredients:
            new_ingredients_list = [ingredient.strip() for ingredient in new_ingredients.split(',')]
            st.session_state.ingredients.extend(new_ingredients_list)
            st.session_state.ingredients = list(set(st.session_state.ingredients))
            st.rerun()  # UI 업데이트

# 사용자 건강 상태와 땡기는 음식 입력 받기
health_condition = st.text_input("Enter your health condition (e.g., diabetes, hypertension, etc.)", value="")
craving_food = st.text_input("What food are you craving right now?", value="")

# Analyze 버튼
if st.button("Analyze and Get Recipes"):
    if st.session_state.ingredients:
        gpt_response = generate_recipe_response(st.session_state.ingredients, health_condition, craving_food)
        recipes = parse_recipes(gpt_response)

        st.write("### Recommended Recipes")

        cols = st.columns(3)  # 3개의 열로 카드 형식의 레이아웃 생성

        for i, recipe in enumerate(recipes):
            with cols[i % 3]:
                st.subheader(recipe["name"])
                st.write(f"**Cooking time:** {recipe['cooking_time']}")
                st.write(f"**Additional ingredients:** {recipe['additional_ingredients']}")

                # Expander 사용하여 준비 단계 표시
                with st.expander("Show Preparation Steps"):
                    st.write(recipe["steps"])

    else:
        st.write("Please upload an image and specify ingredients.")