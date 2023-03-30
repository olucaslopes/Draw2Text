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

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)


if st.sidebar.button('Save Image'):
    save_png(canvas_result.image_data)

    # with open('current_image_array.npy', 'wb') as f:
    #     np.save(f, canvas_result.image_data)

    fig = predict_drawings(canvas_result.image_data)
    st.pyplot(fig=fig)




