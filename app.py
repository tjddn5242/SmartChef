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
import replicate
import numpy as np

# Image FLUX AI 
REPLICATE_API_TOKEN = st.secrets['REPLICATE_API_TOKEN']

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
    with st.spinner("🥕AI 쉐프가 재료를 확인하고 있어요!🥕"):
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
                        {"type": "text", "text": "입력받은 냉장고 속 이미지에서 확실하게 보이는 식재료들만 리스트로 뽑아줘. 이때 식재료와 관련한 이모지를 같이 붙여줘. 불필요한 설명은 제외. format example : ['🥚계란','🎃호박','🍎사과']. 인식된 재료가 없을 경우 빈 리스트를 반환해줘."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        ingredients_list = response.json()['choices'][0]['message']['content']
        ingredients_list = ast.literal_eval(ingredients_list)
        
    st.success("냉장고 재료 인식이 끝났습니다!")
    return ingredients_list


# 음성생성
def generate_and_play_speech(voice, text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,  # options: alloy, echo, fable, onyx, nova, shimmer
        input=text
    )

    output_file_path = "output.wav"
    response.stream_to_file(output_file_path)

    # Streamlit playback
    st.audio(output_file_path)

# Streamlit 앱 설정
st.set_page_config(page_title="Smart Fridge Recipe Recommender", page_icon="🍽️", layout="wide")

# 제목과 스타일링
st.markdown("<h1 style='text-align: center; color: #FF6347;'>스마트쉐프</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FF4500;'>냉장고에 있는 재료로 최고의 음식을 만들어드립니다</p>", unsafe_allow_html=True)

# 이미지 업로드 기능
st.markdown("### 1. 냉장고 사진을 업로드 혹은 직접 촬영 해주세요")
# 사용자가 선택할 수 있는 드롭다운 메뉴 제공
option = st.selectbox("이미지 입력 방식을 선택하세요:", ('파일 업로드', '사진 촬영'))

img_file = None

if option == '파일 업로드':
    img_file = st.file_uploader("찍어놓은 냉장고 내부 사진을 업로드하세요", type=["jpg", "jpeg", "png"])
elif option == '사진 촬영':
    img_file = st.camera_input("사진을 찍어주세요")

# if img_file is not None:
#     st.image(img_file, caption='선택된 이미지')

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
        # detected_ingredients = ["🥔감자", "🥚달걀", "🫑파프리카", "🥒오이", "🌶️고추", "🥕당근"] # 디버깅용 ===========================================================================
        # st.write("Recognized Ingredients:")
        # st.write(detected_ingredients) # 디버깅용 ===========================================================================
        st.session_state.ingredients = list(set(detected_ingredients))
    
        # Detected Ingredients Display (5 items per row)
        st.markdown("### 2. 인식된 재료들을 확인해보세요.")

    # Ensure the session state is set up correctly
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = ingredients

    if 'remove_indices' not in st.session_state:
        st.session_state.remove_indices = []

    if 'new_ingredients_list' not in st.session_state:
        st.session_state.new_ingredients_list = []

    # 재료 삭제 기능
    with st.expander("각 재료 옆의 체크박스를 선택하고 삭제 확정하기 버튼을 눌러 잘못 인식된 재료들을 삭제 할 수 있습니다.", expanded=True):
        if st.session_state.ingredients:
            cols = st.columns(6)  # 한 번만 열을 생성합니다.
            for idx, ingredient in enumerate(st.session_state.ingredients):
                col = cols[idx % 6]  # 현재 재료의 인덱스에 맞는 열을 선택합니다.
                with col:
                    # 하나의 컨테이너에 ingredient와 체크박스를 함께 담기
                    container = st.container(border=True)
                    with container:
                        col1, col2 = st.columns([6, 1])
                        col1.markdown(f"<p style='font-size:20px; text-align:left;'>{ingredient}</p>", unsafe_allow_html=True)
                        if col2.checkbox('', key=f"remove_{ingredient}_{idx}"):
                            if idx not in st.session_state.remove_indices:
                                st.session_state.remove_indices.append(idx)
        else:
            st.markdown("<p style='font-size:16px;'>No ingredients detected yet. Please upload an image.</p>", unsafe_allow_html=True)
    
    # 삭제 버튼
    if st.button("삭제 확정하기"):
        # 삭제할 인덱스를 역순으로 정렬 후 pop으로 삭제하여 인덱스 오류 방지
        for idx in sorted(st.session_state.remove_indices, reverse=True):
            removed_ingredient = st.session_state.ingredients.pop(idx)
            st.toast(f"{removed_ingredient} 이(가) 최종적으로 삭제되었습니다.", icon="🗑️")
            time.sleep(1)
            st.rerun()

    # 재료 추가 기능
    st.markdown("### 3. 인식하지 못한 재료들을 입력해서 추가해보세요.")
    new_ingredients = st.text_input("여러 재료를 입력할 때는 콤마(,)를 이용해주세요", placeholder="ex. 감자, 숙주, 소세지")
    if st.button("재료 추가하기"):
        if new_ingredients:
            new_ingredients_list = [ingredient.strip() for ingredient in new_ingredients.split(',')]
            st.session_state.new_ingredients_list.extend(new_ingredients_list)
            st.session_state.new_ingredients_list = list(set(st.session_state.new_ingredients_list))
            st.toast(f"{', '.join(new_ingredients_list)} 재료(들)이가 추가되었습니다.", icon="✅")
            time.sleep(1)
            # st.rerun()

            # 새로운 재료 추가
            if st.session_state.new_ingredients_list:
                st.session_state.ingredients.extend(st.session_state.new_ingredients_list)
                st.session_state.ingredients = list(set(st.session_state.ingredients))
                st.rerun()

        # 변경 후 상태 초기화
        st.session_state.remove_indices.clear()
        st.session_state.new_ingredients_list.clear()

        # UI 업데이트
        st.rerun()

    # 사용자 건강 상태와 땡기는 음식 입력 받기
    st.markdown("### 4. 가지고 있는 질병과 현재 땡기는 음식을 말씀해주세요.")
    health_condition = st.text_input("가지고 있는 질병이 있다면 입력해주세요 (ex. 당뇨병, 야맹증, 고혈압 등)", placeholder="필수로 입력해주세요")
    craving_food = st.text_input("지금 땡기는 음식이 있다면 입력해주세요", placeholder="없다면 입력하지 않으셔도 됩니다")

    # Analyze 버튼
    if st.button("음식을 추천해줘", help="Click to find recipes based on your ingredients and preferences"):
        with st.spinner('👨‍🍳AI 쉐프가 당신의 건강에 맞는 음식을 찾고 있어요!👨‍🍳'):
            if st.session_state.ingredients:

                gpt_response = json.loads(gptOutput(craving_food, st.session_state.ingredients, health_condition)[0])
                health_summary = gpt_response['chefTip']
                recipes = gpt_response['recipes']

                # print(gpt_response)
                # print(health_summary)
                # print(recipes)

                # 건강 요약 부분을 별도로 출력
                if health_summary:
                    st.markdown("### AI 영양사 한마디")
                    st.markdown(f"**{health_summary}**")
                    st.markdown("---")  # 구분선을 추가하여 건강 요약과 레시피를 구분

                    # Example usage
                    # Assuming you have a client object already created
                    voice = "nova"
                    text = health_summary
                    generate_and_play_speech(voice, text)

                st.markdown("### 추천 레시피")

                cols = st.columns(3)  # 3개의 열로 카드 형식의 레이아웃 생성

                for i, recipe in enumerate(recipes.values()):
                    with cols[i % 3]:
                        st.markdown(f"<h3 style='color: #FF4500;'>{recipe['name']}<br>건강점수: {recipe['health_score']}</h3>", unsafe_allow_html=True)
                        # st.image('https://oaidalleapiprodscus.blob.core.windows.net/private/org-tCAIJLieoZ5a5hHAL85SpD2O/user-oAmOYDR8Wvv7i718IYxSkOyy/img-DDzRwXOBZ09QPNBYWC1RXJ7N.png?st=2024-09-01T08%3A08%3A37Z&se=2024-09-01T10%3A08%3A37Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-08-31T23%3A39%3A59Z&ske=2024-09-01T23%3A39%3A59Z&sks=b&skv=2024-08-04&sig=/Jhnj1DHBkkL/OJSpAzkAUpZ87AAoBKRseDT1qrDpEc%3D', caption='Your image caption', use_column_width=True)
                        st.markdown(f"조리시간: {recipe['cooking_time']}")
                        st.markdown(f"필요재료: {recipe['all_ingredients']}")
                        st.markdown(f"추가구비재료: {recipe['additional_ingredients']}")

                        input = {
                            "prompt": f"Realistically, {recipe['english_name']}, and Korean style food, Only Food, tasty, dynamic shot"
                        }

                        output = replicate.run(
                            "black-forest-labs/flux-schnell",
                            input=input
                        )

                        # Expander 사용하여 준비 단계 표시
                        with st.expander("조리방법보기"):
                            st.markdown("#### 조리 방법")
                            # 조리 단계에서 줄바꿈 적용하여 표시
                            steps = recipe['steps'].split('\n')
                            for step in steps:
                                st.markdown(f"{step.strip()}")

                        st.image(output[0], output_format="JPEG")

else:
    st.warning("먼저 사진을 업로드 해주세요")

# st.write(type(recipes))
# st.write(recipes)

# user_need = craving_food
# ingredients = st.session_state.ingredients
# disease = health_condition

# st.markdown(f'''
# ### 변수
# - user_need: {user_need}
# - ingredients: {ingredients}
# - disease: {disease}
# ---
# ### 프롬프트
# {gptOutput(user_need, ingredients, disease)[1]}
# ---
# ### 모델output
# {gptOutput(user_need, ingredients, disease)[0]}
# ''')
