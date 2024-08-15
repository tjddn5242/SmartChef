import streamlit as st
from PIL import Image
import openai
from transformers import CLIPProcessor, CLIPModel
import torch

# OpenAI API Key 설정

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 사용자 정의 레시피 데이터
recipes = {
    "salad": {
        "ingredients": ["lettuce", "tomato", "cucumber", "olive oil"],
        "steps": ["Chop the lettuce, tomato, and cucumber.", "Mix them in a bowl.", "Add olive oil and mix well."]
    },
    "smoothie": {
        "ingredients": ["banana", "strawberry", "yogurt", "honey"],
        "steps": ["Peel and slice the banana.", "Combine banana, strawberries, yogurt, and honey in a blender.", "Blend until smooth."]
    }
}

# 확장된 재료 리스트
ingredient_list = [
    "lettuce", "tomato", "cucumber", "olive oil", "banana", "strawberry", "yogurt", "honey",
    "cheese", "bread", "egg", "chicken", "beef", "pork", "fish", "garlic", "onion", "carrot",
    "potato", "bell pepper", "spinach", "mushroom", "avocado", "rice", "pasta", "milk", "butter",
    "flour", "sugar", "salt", "pepper", "chocolate", "bacon", "sausage", "apple", "orange", "grapes",
    "peanut butter", "almond", "walnut", "blueberry", "raspberry", "blackberry", "cabbage", "zucchini"
]

# 이미지에서 재료 인식 함수
def recognize_ingredients_from_image(image):
    inputs = processor(text=ingredient_list, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 이미지에 대한 텍스트의 로짓
    probs = logits_per_image.softmax(dim=1)  # 확률 계산

    # 인식된 재료를 반환
    recognized_ingredients = []
    threshold = 0.05  # 예제 기준, 재료 인식 확률이 5% 이상일 경우
    for i, prob in enumerate(probs[0]):
        if prob > threshold:
            recognized_ingredients.append(ingredient_list[i])

    return recognized_ingredients

# 레시피 추천 함수
def generate_recipe_response(ingredients, health_condition):
    available_recipes = []
    for recipe_name, recipe_details in recipes.items():
        if all(item in ingredients for item in recipe_details["ingredients"]):
            available_recipes.append(recipe_name)

    if not available_recipes:
        return "Sorry, no suitable recipes found for your condition."

    prompt = f"Based on the ingredients: {', '.join(ingredients)} and your health condition: {health_condition}, I recommend the following recipes: {', '.join(available_recipes)}.\n\nHere are the detailed steps for the recipes:\n"

    for recipe_name in available_recipes:
        steps = "\n".join(recipes[recipe_name]["steps"])
        prompt += f"\nRecipe: {recipe_name}\nIngredients: {', '.join(recipes[recipe_name]['ingredients'])}\nSteps:\n{steps}\n"

    # OpenAI GPT-4 API를 사용하여 레시피 추천을 생성합니다.
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit 앱 설정
st.title("Smart Fridge Recipe Recommender")

# 초기 상태 설정
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = []

# 이미지 업로드 기능
img_file = st.file_uploader("Upload a picture of your fridge", type=["jpg", "jpeg", "png"])

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
            # 쉼표로 구분된 입력을 리스트로 변환하고 중복을 제거한 후 추가
            new_ingredients_list = [ingredient.strip() for ingredient in new_ingredients.split(',')]
            st.session_state.ingredients.extend(new_ingredients_list)
            st.session_state.ingredients = list(set(st.session_state.ingredients))
            st.rerun()  # UI 업데이트

# 사용자 건강 상태 입력 받기
health_condition = st.selectbox("Select your health condition", ["None", "diabetes", "hypertension"])

# Analyze 버튼
if st.button("Analyze and Get Recipes"):
    if st.session_state.ingredients and health_condition:
        recommendation = generate_recipe_response(st.session_state.ingredients, health_condition)
        st.write("Recommendation:", recommendation)
    else:
        st.write("Please upload an image and select a health condition.")