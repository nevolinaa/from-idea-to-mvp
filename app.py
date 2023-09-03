import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict

def process_main_page():
    show_main_page()
    process_side_bar_inputs()

def show_main_page():
    image = Image.open('./data/family.png')

    st.set_page_config(
        page_icon = "🌎",
        layout="centered",
        initial_sidebar_state="collapsed",
        page_title="SatisFlight",
    )

    st.write(
        """
        # Узнайте, доволен ли Ваш пассажир перелетом
        """
    )

    st.image(image)

def write_user_data(df):
    st.write("## Полученные данные ")
    st.write(df)

def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание:")
    st.write(prediction)

    st.write("## Вероятность предсказания:")
    st.write(prediction_probas)

def process_side_bar_inputs():
    user_input_df = sidebar_input_features()

    if (st.button('Готово')):
        train_df = open_data()
        train_X_df, _ = split_data(train_df)
        full_X_df = pd.concat((user_input_df, train_X_df), axis=0, ignore_index=True)
        preprocessed_X_df = preprocess_data(full_X_df, test=False)

        user_X_df = preprocessed_X_df[:1]
        write_user_data(user_X_df)

        prediction, prediction_probas = load_model_and_predict(user_X_df)
        write_prediction(prediction, prediction_probas)
        
        st.divider()
        st.subheader("Справка")
        st.markdown(
            """
            Предсказания сделаны на основе **81038** наблюдений, среди которых:
            - **45%** довольных перелетом пассажиров
            - **82%** лояльных к компании пассажиров
            - **48%** пассажиров бизнес-класса
            - **30%** пассажиров, столкнувшихся с задержкой рейса
            """
        )

def sidebar_input_features():
    st.write("### Введите информацию о пассажире ")

    type = st.selectbox("Пассажир лоялен к авиакомпании?", ('Да', 'Нет'))

    pclass = st.selectbox("Класс обслуживания пассажира", ("Эконом", "Комфорт", "Бизнес"))

    delay = st.selectbox("Были ли задержки в прибытии рейса?", ("Да", "Нет"))

    checkin = st.select_slider('Оценка пассажиром сервиса регистрации на рейс, где 1 – это «плохо», а 5 – это «отлично»',
                               options=[1, 2, 3, 4, 5])

    on_board = st.select_slider('Оценка пассажиром обслуживания на борту, где 1 – это «плохо», а 5 – это «отлично»',
                             options=[1, 2, 3, 4, 5])

    seat_comfort = st.select_slider('Оценка пассажиром удобства сиденья, где 1 – это «плохо», а 5 – это «отлично»',
                               options=[1, 2, 3, 4, 5])

    cleanliness = st.select_slider('Оценка пассажиром чистоты на борту, где 1 – это «плохо», а 5 – это «отлично»',
                                    options=[1, 2, 3, 4, 5])

    translatetion = {
        "Да": 1,
        "Нет": 0,
        "Бизнес": "Business",
        "Комфорт": "Eco Plus",
        "Эконом": "Eco"
    }

    data = {
        'Customer Type': translatetion[type],
        'On-board service': on_board,
        'Checkin service': checkin,
        'Seat comfort': seat_comfort,
        'Cleanliness': cleanliness,
        'Class': translatetion[pclass],
        'Any Arrival Delay': translatetion[delay]
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
