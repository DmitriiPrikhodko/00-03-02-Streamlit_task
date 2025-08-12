import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yfinance as yf
import math
import numpy as np
import io
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")  # Игнорируем warnings

# Название

st.title(":blue[Задание 00-03-02]")
st.write(
    """
#### _Приходько Дмитрий_
## **Исследование датафрейма "Tips"**
"""
)

# Файл можно загрузить отсюда path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
# path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
# tips = pd.read_csv(path)

st.sidebar.write(
    "Датафрейм взят с https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)


@st.cache_data
def upload_file(file):
    df = pd.read_csv(file, encoding="latin1")
    return df


def save_fig(fig, button_label="Скачать график", filename="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.sidebar.download_button(
        label=button_label,
        data=buf,
        file_name=filename,
        mime="image/png",
    )
    return None


get_file = st.sidebar.file_uploader("Загрузи CSV файл tips.csv")
if get_file is not None:
    tips = upload_file(get_file)
else:
    st.stop()

# Добавляем случайные даты

date_range = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
random_dates = np.random.choice(date_range, size=len(tips.index))
tips["time_order"] = pd.Series(random_dates)

st.write('#### Датафрейм "Tips"')
st.dataframe(tips)

# Строим график чаевых от времени

st.write("## Графики")


tips_vs_time = tips.groupby(pd.Grouper(key="time_order", freq="1D"))["tip"].sum()


fig1 = st.line_chart(data=tips_vs_time, x_label="Дата", y_label="Сумма чаевых ($)")

st.write("##### _График 1._ Чаевые в зависимости от даты")
st.write("______")

# Строим гистограмму распределения величины счета


fig2, ax2 = plt.subplots()

ax2.hist(tips["total_bill"], bins=30)
ax2.set_xlabel("Размер счета ($)")
ax2.set_ylabel("Количество")
st.pyplot(fig2, clear_figure=True)


save_fig(fig2, "Скачать гистограмму (график 2)", "2_bill_size_distrib.png")
st.write("##### _График 2._ Распределение величины счета")
st.write("______")


# Строим зависимость чаевых от величины счета


fig3, ax3 = plt.subplots()

ax3 = sns.scatterplot(tips, x="total_bill", y="tip")
ax3.set_xlabel("Размер счета ($)")
ax3.set_ylabel("Чаевые ($)")
st.pyplot(fig3, clear_figure=True)
save_fig(fig3, "Скачать график 3", "3_tips_vs_bill.png")
st.write("##### _График 3._ Зависимость чаевых от величины счета")
st.write("______")


# Строим зависимость чаевых от величины счета с учетом количества гостей


fig4 = sns.relplot(
    tips,
    x="total_bill",
    y="tip",
    kind="line",
    hue="size",
    palette="muted",
).figure

st.pyplot(fig4)

save_fig(fig4, "Скачать график 4", "4_tips_vs_bill_vs_guest_count.png")
st.write(
    """
         ##### _График 4._ Зависимость чаевых от величины счета
         ##### с учетом количества гостей"""
)
st.write("______")

# Строим зависимость величины счета от дня недели

fig5, ax5 = plt.subplots()
ax5 = sns.swarmplot(tips, x="day", y="total_bill")
ax5.set_xlabel("Размер счета ($)")
ax5.set_ylabel("Количество")

st.pyplot(fig5)

save_fig(fig5, "Скачать график 5", "5_bill_vs_day.png")

st.write("##### _График 5._ Зависимость величины счета от дня недели")
st.write("______")

# Строим зависимость величины чаевых от дня недели c учетом пола


fig6, ax6 = plt.subplots()
ax6 = sns.swarmplot(tips, y="tip", x="day", hue="sex", palette="muted")
ax6.set_xlabel("День недели")
ax6.set_ylabel("Размер чаевых ($)")

st.pyplot(fig6)

save_fig(fig6, "Скачать график 6", "6_bill_vs_day__vs_sex.png")
st.write("##### _График 6._ Зависимость величины чаевых от дня недели с учетом пола")
st.write("______")

# Строим зависимость величины счета от дня недели с учетом времени дня

fig7, ax7 = plt.subplots()
ax7 = sns.boxplot(data=tips, x="day", y="total_bill", hue="time", palette="muted")
ax7.set_xlabel("День недели")
ax7.set_ylabel("Величина счета ($)")

st.pyplot(fig7)

save_fig(fig7, "Скачать график 7", "7_bill_vs_day__vs_time.png")
st.write(
    "##### _График 7._ Зависимость величины счета от дня недели с учетом времени дня"
)
st.write("______")

# Строим распределение величины чаевых c учетом времени дня

fig8 = make_subplots(rows=1, cols=2, horizontal_spacing=0.15)
cols = {"Dinner": 1, "Lunch": 2}
for time, tt in tips.groupby("time"):
    if time == "Dinner":
        time_rus = "Ужин"
    else:
        time_rus = "Обед"
    fig8.add_trace(
        go.Histogram(x=tt["tip"], name=time_rus, nbinsx=40),
        row=1,
        col=cols[time],  # type: ignore
    )

fig8.update_layout(
    xaxis=dict(
        title="Total bill ($)",
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        ticks="outside",
        tickwidth=2,
    ),
    yaxis=dict(
        title="Tips ($)",
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        ticks="outside",
        tickwidth=2,
    ),
    plot_bgcolor="white",
    showlegend=True,
)

fig8.update_xaxes(
    title="Чаевые ($)",
    title_font=dict(size=18, color="black", family="Arial"),
    tickfont=dict(size=14, color="black", family="Arial"),
    showline=True,
    linewidth=2,
    linecolor="black",
    showgrid=True,
    gridcolor="lightgray",
    gridwidth=1,
    ticks="outside",
    tickwidth=2,
    row=1,
    col=1,
)
fig8.update_yaxes(
    title="Количество",
    title_font=dict(size=18, color="black", family="Arial"),
    tickfont=dict(size=14, color="black", family="Arial"),
    showline=True,
    linewidth=2,
    linecolor="black",
    showgrid=True,
    gridcolor="lightgray",
    gridwidth=1,
    ticks="outside",
    tickwidth=2,
    row=1,
    col=1,
)

fig8.update_xaxes(
    title="Чаевые ($)",
    title_font=dict(size=18, color="black", family="Arial"),
    tickfont=dict(size=14, color="black", family="Arial"),
    showline=True,
    linewidth=2,
    linecolor="black",
    showgrid=True,
    gridcolor="lightgray",
    gridwidth=1,
    ticks="outside",
    tickwidth=2,
    row=1,
    col=2,
)
fig8.update_yaxes(
    title="Количество",
    title_font=dict(size=18, color="black", family="Arial"),
    tickfont=dict(size=14, color="black", family="Arial"),
    showline=True,
    linewidth=2,
    linecolor="black",
    showgrid=True,
    gridcolor="lightgray",
    gridwidth=1,
    ticks="outside",
    tickwidth=2,
    row=1,
    col=2,
)

st.plotly_chart(fig8)
st.write("##### _График 8._ Распределение чаевых с учетом времени дня")
st.write("______")


# Строим зависимость величины чаевых от счета c учетом пола и курнеия

fig9 = sns.relplot(
    tips, x="total_bill", y="tip", kind="scatter", col="sex", hue="smoker", height=4
)
fig9.axes_dict["Male"].set_xlabel("Счёт, $")
fig9.axes_dict["Female"].set_xlabel("Счёт, $")
fig9.axes_dict["Male"].set_title("Мужчины")
fig9.axes_dict["Female"].set_title("Женщины")
fig9.set_axis_labels("Счёт ($)", "Чаевые ($)")
fig9._legend.set_title("Курение") # type: ignore
st.pyplot(fig9.figure)


save_fig(fig9, "Скачать график 9", "9_tip_vs_bill__vs_sex/smoke.png")
st.write("##### _График 9._ Зависимость чаевых от счета с учетом пола и курения")
st.write("______")

# Строим корреляционную матрицу чаевых, счета и количества гостей за столом

tips_nums = tips.loc[::, ["total_bill", "tip", "size"]].copy()
tips_nums.columns = ["Величина счета", "Чаевые", "Кол-во человек"]
corr_matrix = pd.DataFrame.corr(tips_nums)

fig10, ax10 = plt.subplots()

ax10 = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
ax10.tick_params(left=False, bottom=False)
ax10.grid(False)
st.pyplot(fig10)
save_fig(fig10, "Скачать график 10", "10_bill_tip_size_corr_matr.png")
st.write(
    "##### _График 10._ Корреляционная матрица чаевых, счета и количества гостей за столом"
)
st.write("______")
