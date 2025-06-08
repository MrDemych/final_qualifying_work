# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Конфигурация страницы
st.set_page_config(page_title="Predictive Maintenance", layout="wide")


# Главная страница
def main_page():
    st.title("Система предиктивного обслуживания оборудования")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Предпросмотр данных:", data.head())

        # Обучение модели
        if st.button("Обучить модель"):
            with st.spinner("Обучение..."):
                # Здесь должен быть код обучения из main.py
                st.success("Модель успешно обучена!")

                # Отображение метрик
                st.subheader("Результаты оценки")
                col1, col2 = st.columns(2)
                with col1:
                    st.image("Logistic_Regression_conf_matrix.png")
                with col2:
                    st.image("Random_Forest_conf_matrix.png")

        # Предсказания
        st.subheader("Прогнозирование отказов")
        sample = data.iloc[0:1].drop('Machine failure', axis=1, errors='ignore')
        if st.button("Предсказать"):
            model = joblib.load("Random_Forest.pkl")
            prediction = model.predict(sample)
            st.write(f"Прогноз: {'Отказ' if prediction[0] else 'Исправен'}")


# Страница презентации
def presentation():
    st.title("Описание проекта")
    st.write("""
    ## Бинарная классификация для предиктивного обслуживания
    **Цель проекта:** Прогнозирование отказов промышленного оборудования
    """)
    st.image("architecture.png", caption="Архитектура решения")


# Навигация
pages = {
    "Главная страница": main_page,
    "Презентация": presentation
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Выберите страницу", list(pages.keys()))
pages[selection]()