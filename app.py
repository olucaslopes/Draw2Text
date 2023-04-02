import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils import predict_drawings, save_png


st.set_page_config(
    page_title="Draw2Text", page_icon="👁️"
)

st.markdown('''
# Draw2Text

AI-powered drawing recognition app that uses advanced machine learning algorithms to detect\
numbers and letters in drawing sketches

''')

# Specify canvas parameters in application
drawing_mode = "freedraw"

stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 3)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#000000',
    background_color='#EEEEEE',
    update_streamlit=True,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)


if st.sidebar.button('Predict Digits'):
    save_png(canvas_result.image_data)

    # with open('current_image_array.npy', 'wb') as f:
    #     np.save(f, canvas_result.image_data)

    fig = predict_drawings(canvas_result.image_data)
    st.pyplot(fig=fig)
