import streamlit as st
from itertools import cycle
from streamlit_drawable_canvas import st_canvas
import streamlit_toggle as tog
from utils import predict_drawings, save_png
from PIL import Image, ImageOps
import numpy as np
from exceptions import DigitNotFoundError


st.set_page_config(
    page_title="Draw2Text", page_icon="👁️"
)

st.markdown('''
# 👁️ Draw2Text

AI-powered drawing recognition app that uses advanced machine learning algorithms to detect\
numbers and letters in drawing sketches

''')

stroke_slider = 3

if 'pressed_predict_button' not in st.session_state:
    st.session_state.pressed_predict_button = False

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_slider,
    stroke_color='#000000',
    background_color='#EEEEEE',
    update_streamlit=True,
    height=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    pil_image = Image.fromarray(canvas_result.image_data, mode='RGBA')
    gray_img = ImageOps.grayscale(pil_image)
    img_array = np.array(gray_img)
    if np.all(img_array == 238):
        empty_canvas = True
        st.session_state.pressed_predict_button = False
    else:
        empty_canvas = False
else:
    empty_canvas = True
    st.session_state.pressed_predict_button = False

colc1, colc2, colc3 = st.columns([2, 3, 3])

colc1.text('\n')
colc1.text('\n')
predict_button = colc1.button('Predict Digits', disabled=empty_canvas)

stroke_slider = colc2.slider("Stroke width: ", 1, 10, 3)

colc3.empty()


if (predict_button and not empty_canvas) or st.session_state.pressed_predict_button:

    st.session_state.pressed_predict_button = True

    try:
        digits, predicted_labels = predict_drawings(canvas_result.image_data)
    except DigitNotFoundError:
        pass
    else:
        save_png(canvas_result.image_data)
        st.markdown('#### Found digits:')

        tog.st_toggle_switch(label='Edit mode')
        caption = [f'predicted={p}' for p in predicted_labels]
        cols = cycle(st.columns(5))  # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(digits):
            current_col = next(cols)
            digits[0].save('test.png')
            current_col.image(ImageOps.invert(filteredImage), width=100)
            current_col.text(caption[idx])
