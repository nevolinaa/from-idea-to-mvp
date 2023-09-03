# Проект в рамках интенсива «Разработка ML сервиса: от идеи к прототипу»

Интенсив включает в себя проведение разведочного анализа данных, построение моделей машинного обучения и создание интерактивного приложения на их основе для презентации результатов

[Интерактивное приложение](https://satisflight.streamlit.app/)

### Что в репозитории?
- Используемые данные и модель с подобранными параметрами - в папке data

- Разведочный анализ данных и построение моделей машинного обучения - EDA_ML.ipynb 
- Приложение - app.py
- Используемые в приложении функции и модель - model.py
- Необходимые для работы приложения библиотеки и их версии - requirements.txt

### Описание данных

**Датасет содержит информацию о клиентах некоторой авиакомпании.**  
_Целевая переменная_ (таргет) – `satisfaction` (удовлетворенность клиента полетом), бинарная (_satisfied_ или _neutral or dissatisfied_)

<img src="data/family.png" width= 60%  />

**Признаки**
- `Gender` (categorical: _Male_ или _Female_): пол клиента
- `Age` (numeric, int): количество полных лет
- `Customer Type` (categorical: _Loyal Customer_ или _disloyal Customer_): лоялен ли клиент авиакомпании?
- `Type of Travel` (categorical: _Business travel_ или _Personal Travel_): тип поездки
- `Class` (categorical: _Business_ или _Eco_, или _Eco Plus_): класс обслуживания в самолете
- `Flight Distance` (numeric, int): дальность перелета (в милях)
- `Departure Delay in Minutes` (numeric, int): задержка отправления (неотрицательная)
- `Arrival Delay in Minutes` (numeric, int): задержка прибытия (неотрицательная)

**Признаки, перечисленные ниже, являются числовыми. По смыслу они категориальные: клиент ставил оценку от 1-го до 5-ти включительно**
- `Inflight wifi service` (categorical, int): оценка клиентом интернета на борту
- `Departure/Arrival time convenient` (categorical, int): оценка клиентом удобство времени прилета и вылета
- `Ease of Online booking` (categorical, int): оценка клиентом удобства онлайн-бронирования
- `Gate location` (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту
- `Food and drink` (categorical, int): оценка клиентом еды и напитков на борту
- `Online boarding` (categorical, int): оценка клиентом выбора места в самолете
- `Seat comfort` (categorical, int): оценка клиентом удобства сиденья
- `Inflight entertainment` (categorical, int): оценка клиентом развлечений на борту
- `On-board service` (categorical, int): оценка клиентом обслуживания на борту
- `Leg room service` (categorical, int): оценка клиентом места в ногах на борту
- `Baggage handling` (categorical, int): оценка клиентом обращения с багажом
- `Checkin service` (categorical, int): оценка клиентом регистрации на рейс
- `Inflight service` (categorical, int): оценка клиентом обслуживания на борту
- `Cleanliness` (categorical, int): оценка клиентом чистоты на борту

### Этапы проекта
- EDA (работа с пропусками и выбросами, кодирование и масштабирование признаков, анализ корреляции признаков с таргетом и между собой, построение графиков распределения признаков)
- Построение моделей машинного обучения (Logistic Regression, Decicion Tree, Random Forest, KNN) и подбор гиперпараметров
- Сравнение метрик качества (accuracy, precision, recall, AUC ROC, F1-score) и выбор лучшей модели
- Создание интерактивного приложения на основе имеющихся данных и модели
