import pandas as pd
import streamlit as st
from itertools import cycle
from streamlit_drawable_canvas import st_canvas
from utils import predict_drawings, save_png
from PIL import ImageOps


st.set_page_config(
    page_title="Draw2Text", page_icon="👁️"
)

st.markdown('''
# 👁️ Draw2Text

AI-powered drawing recognition app that uses advanced machine learning algorithms to detect\
numbers and letters in drawing sketches

''')

stroke_slider = 3

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

colc1, colc2, colc3 = st.columns([2 ,3, 3])

colc1.text('\n')
colc1.text('\n')
predict_button = colc1.button('Predict Digits')

stroke_slider = colc2.slider("Stroke width: ", 1, 10, 3)

colc3.empty()


if predict_button:
    save_png(canvas_result.image_data)

    # with open('current_image_array.npy', 'wb') as f:
    #     np.save(f, canvas_result.image_data)

    st.markdown('#### Founded Digits:')

    digits, predicted_labels = predict_drawings(canvas_result.image_data)

    caption = [f'predicted={p}' for p in predicted_labels]
    cols = cycle(st.columns(5))  # st.columns here since it is out of beta at the time I'm writing this
    for idx, filteredImage in enumerate(digits):
        current_col = next(cols)
        digits[0].save('test.png')
        current_col.image(ImageOps.invert(filteredImage), width=100)
        current_col.text(caption[idx])
