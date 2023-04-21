import streamlit as st
from itertools import cycle
from streamlit_drawable_canvas import st_canvas
from utils import predict_drawings, save_digit
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
predict_button = colc1.button('Predict Digits', disabled=empty_canvas, type='primary')

# stroke_slider = colc2.slider("Stroke width: ", 1, 10, 3)

colc3.empty()


if (predict_button and not empty_canvas) or st.session_state.pressed_predict_button:

    st.session_state.pressed_predict_button = True

    try:
        if predict_button:
            digits, predicted_labels = predict_drawings(canvas_result.image_data)
            st.session_state.digits = digits
            st.session_state.predicted_labels = predicted_labels
        else:
            digits = st.session_state.digits
            predicted_labels = st.session_state.predicted_labels
    except DigitNotFoundError:
        pass
    else:
        # save_png(canvas_result.image_data)
        st.markdown('#### Found digits:')

        edit_mode = st.checkbox('Toggle label editing')

        caption = [f'predicted={p}' for p in predicted_labels]

        if edit_mode:
            opt = [i for i in range(0, 10)] + ['Undefiable', ]
            for idx, filteredImage in enumerate(digits):
                cols = st.columns(5)
                cols[1].image(ImageOps.invert(filteredImage), width=100, caption=caption[idx])
                cols[2].selectbox(
                    options=opt,
                    index=10,
                    label='Select the true label',
                    key=f'y_true_{idx}'
                )

            feedback_completed = st.button('Send')

            if feedback_completed:
                for idx, d in enumerate(digits):
                    save_digit(d,
                               metadata={
                                   'y_true': st.session_state[f'y_true_{idx}'],
                                   'y_pred': predicted_labels[idx]}
                               )
                st.success('Thanks for the feedback!', icon="✅")
        else:
            cols = cycle(st.columns(5))  # st.columns here since it is out of beta at the time I'm writing this
            for idx, filteredImage in enumerate(digits):
                current_col = next(cols)
                current_col.image(ImageOps.invert(filteredImage), width=100, caption=caption[idx])
