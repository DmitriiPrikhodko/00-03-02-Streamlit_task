import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yfinance as yf
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")  # Игнорируем warnings

# Название

st.title(":blue[Задание 00-03-02]")
st.write(
    """
#### _Приходько Дмитрий_
## **Стоимость акций компании Apple**
"""
)

plt.rcParams["figure.dpi"] = 150
plt.style.use("bmh")

tickerSymbol = "AAPL"

tickerData = yf.Ticker(tickerSymbol)

tickerDF_today = tickerData.history(period="1d", interval="15m")
tickerDF_today.columns = [
    "Открытие",
    "Макс",
    "Мин",
    "Закрытие",
    "Объем",
    "Дивиденты",
    "Сплиты",
]
# tickerDF_today.index = tickerDF_today.index.strftime("%H:%M")
st.write("#### Данные за последний час")
st.write(tickerDF_today.tail(60))
st.write("#### Закрытие от времени за сегодня")
fig1, ax1 = plt.subplots()
ax1.plot(tickerDF_today.index, tickerDF_today["Закрытие"])
ax1.set_xlabel("Время")
ax1.set_ylabel("Цена ($)")
ax1.tick_params(axis="x", labelrotation=45)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

st.pyplot(fig1, clear_figure=True)

tickerDF = tickerData.history(start="2011-01-01", interval="1d")
# tickerDF.index = tickerDF.index.strftime("%d-%m-%Y")
tickerDF.columns = [
    "Открытие",
    "Макс",
    "Мин",
    "Закрытие",
    "Объем",
    "Дивиденты",
    "Сплиты",
]
st.write("#### Данные за последние 30 дней")
st.write(tickerDF.tail(30))
st.write("#### Закрытие от времени (с 01-01-2011)")
# st.line_chart(tickerDF.Закрытие)
fig2, ax2 = plt.subplots()
ax2.autoscale(enable=True, axis='x')
ax2.plot(tickerDF.index, tickerDF["Закрытие"])
ax2.set_xlabel("Время")
ax2.set_ylabel("Цена ($)")
ax2.tick_params(axis="x", labelrotation=45)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
st.pyplot(fig2, clear_figure=True)
