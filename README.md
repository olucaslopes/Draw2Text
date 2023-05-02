# Draw2Text

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://draw2text.streamlit.app/)

<p align="center">
    <a href="https://energygenforecaster.streamlit.app/"><img src="img/Draw2Text-Demo.gif" width="50%" align="center"/></a>
</p>

AI-powered drawing recognition app that uses advanced machine learning algorithms to detect numbers and letters in drawing sketches

## How it works

The app uses computer vision models to segment each of the canvas digits. Then, it applies a neural network to predict the class of each digit (10 classes, from 0 to 9). Finally, it shows to the user each predicted class of the founded digits, and the user can then toggle label editing to send us the correct labels for future tuning.

## Getting Started

To run the app locally, please follow these steps:

1. Clone the repository

```
git clone https://github.com/olucaslopes/Draw2Text.git
```

2. Install the required packages

```
pip install -r requirements.txt
```

3. Run the app

```
streamlit run app.py
```

## Contact

You can find out more about me [on my Linkedin](https://www.linkedin.com/in/o-lucas-lopes)