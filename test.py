from llmStructure import *

# user input
user_need = '뜨끈한 해물이 포함된 국물이 있는 요리.'
ingredients = '오징어| 새우| 어묵| 조개| 홍합| 고추가루'
disease = '당뇨병'

print(gptOutput(user_need, ingredients, disease))