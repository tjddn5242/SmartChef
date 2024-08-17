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

# 영어 재료명과 대응하는 한국어 재료명 사전
ingredient_translation = {
    "lettuce": "상추", "tomato": "토마토", "cucumber": "오이", "olive oil": "올리브 오일",
    "banana": "바나나", "strawberry": "딸기", "yogurt": "요거트", "honey": "꿀",
    "cheese": "치즈", "bread": "빵", "egg": "계란", "chicken": "닭고기", "beef": "소고기",
    "pork": "돼지고기", "fish": "생선", "garlic": "마늘", "onion": "양파", "carrot": "당근",
    "potato": "감자", "bell pepper": "피망", "spinach": "시금치", "mushroom": "버섯",
    "avocado": "아보카도", "rice": "쌀", "pasta": "파스타", "milk": "우유", "butter": "버터",
    "flour": "밀가루", "sugar": "설탕", "salt": "소금", "pepper": "후추", "chocolate": "초콜릿",
    "bacon": "베이컨", "sausage": "소세지", "apple": "사과", "orange": "오렌지", "grapes": "포도",
    "peanut butter": "땅콩버터", "almond": "아몬드", "walnut": "호두", "blueberry": "블루베리",
    "raspberry": "라즈베리", "blackberry": "블랙베리", "cabbage": "양배추", "zucchini": "애호박"
}

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

        # 인식된 재료를 한국어로 변환
        recognized_ingredients_ko = [ingredient_translation.get(ingredient, ingredient) for ingredient in recognized_ingredients]
        return recognized_ingredients_ko
    except Exception as e:
        st.error(f"Error in ingredient recognition: {e}")
        return []

def generate_recipe_response(ingredients, health_condition=None, craving_food=None):
    if not health_condition:
        health_condition = "없음"
    if not craving_food:
        craving_food = "없음"

    prompt = (
        f"""다음 재료들이 있습니다: {', '.join(ingredients)}.
        제 건강 상태는 {health_condition}이고, 현재 {craving_food}을(를) 먹고 싶습니다. 
        이 재료들을 사용하여 만들 수 있는 다양한 요리 레시피를 추천해 주세요. 제 건강 상태와 먹고 싶은 음식을 고려해 주세요.
        
        **중요: 아래 형식에 맞추어 정확히 답변해 주세요. 추가 정보나 텍스트는 추가하지 마세요. 준수할 수 없다면 'N/A'라고 출력해 주세요.**

        <출력 형식>
        요리 이름:
        조리 시간:
        필요재료:
        추가로 구비해야 하는 재료:
        요리 단계:
        """
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative and helpful chef who gives recipes in Korean."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def parse_recipes(gpt_response):
    recipes = []
    recipe_blocks = gpt_response.split("\n\n")  # 공백 두 줄로 레시피 블록을 나눕니다.

    for block in recipe_blocks:
        lines = block.strip().split("\n")
        name = None
        cooking_time = "알 수 없음"
        all_ingredients = "없음"
        additional_ingredients = "없음"
        steps_start = 0

        for i, line in enumerate(lines):
            if "요리 이름:" in line:
                name = line.replace("요리 이름:", "").strip()
            if "조리 시간:" in line:
                cooking_time = line.replace("조리 시간:", "").strip()
            if "필요재료:" in line:
                all_ingredients = line.replace("필요재료:", "").strip()
            if "추가로 구비해야 하는 재료:" in line:
                additional_ingredients = line.replace("추가로 구비해야 하는 재료:", "").strip()
            if "요리 단계:" in line:
                steps_start = i + 1
                break

        if not name:
            name = lines[0].strip()

        steps = "\n".join(lines[steps_start:])
        recipes.append({
            "name": name,
            "cooking_time": cooking_time,
            "all_ingredients": all_ingredients,
            "additional_ingredients": additional_ingredients,
            "steps": steps
        })

    return recipes

# Streamlit 앱 설정
st.set_page_config(page_title="Smart Fridge Recipe Recommender", page_icon="🍽️", layout="wide")

# 제목과 스타일링
st.markdown("<h1 style='text-align: center; color: #FF6347;'>스마트쉐프</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FF4500;'>냉장고에 있는 재료로 최고의 음식을 만들어드립니다</p>", unsafe_allow_html=True)

# 이미지 업로드 기능
st.markdown("### 1. 냉장고 사진을 업로드 해주세요")
img_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

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
        detected_ingredients = recognize_ingredients_from_image(img)
        st.session_state.ingredients = list(set(detected_ingredients))
    
    # Detected Ingredients Display (5 items per row)
    st.markdown("### 2. 인식된 재료들을 확인해보세요.")
    with st.expander("각 재료 옆의 x버튼을 눌러 잘못 인식된 재료들을 삭제 할 수 있습니다.", expanded=True):
        if st.session_state.ingredients:
            rows = len(st.session_state.ingredients) // 5 + 1
            for i in range(rows):
                cols = st.columns(5)
                for j in range(5):
                    idx = i * 5 + j
                    if idx < len(st.session_state.ingredients):
                        ingredient = st.session_state.ingredients[idx]
                        with cols[j]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"<p style='font-size:16px;'>{ingredient}</p>", unsafe_allow_html=True)
                            with col2:
                                if st.button('X', key=f"remove_{ingredient}_{idx}"):
                                    st.session_state.ingredients.pop(idx)
                                    st.rerun()  # UI 업데이트
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
            gpt_response = generate_recipe_response(st.session_state.ingredients, health_condition, craving_food)
            recipes = parse_recipes(gpt_response)

            st.markdown("### 추천 레시피")

            cols = st.columns(3)  # 3개의 열로 카드 형식의 레이아웃 생성

            for i, recipe in enumerate(recipes):
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