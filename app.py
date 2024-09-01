import streamlit as st
import replicate
from IPython.display import Image

REPLICATE_API_TOKEN = st.secrets['REPLICATE_API_TOKEN']

input = {
    "prompt": "Realistically, hot squid soup, and Korean style"
}

output = replicate.run(
    "black-forest-labs/flux-schnell",
    input=input
)

# Image(url=output[0])
st.image(output[0], output_format="JPEG")