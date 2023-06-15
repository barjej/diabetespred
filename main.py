import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout='wide',page_title='Prediksi Diagnostik Penyakit Diabetes')

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRW-2rCgMoo9glcRfoAAZHUIgQpyb4sXv9DATBsQoihLtBbCo-p_BUwfrNcaq8OCCOe2ZhS0mVOvkR-/pub?gid=1879661496&single=true&output=csv')

#########################################################################
#DATA PREPROCESSING

#Handling Outlier
df['Pregnancies']=df.Pregnancies.mask(df.Pregnancies == 0, df['Pregnancies'].mean())
df['Pregnancies']=df.Pregnancies.mask(df.Pregnancies > 12, df['Pregnancies'].mean())
df['Glucose']=df.Glucose.mask(df.Glucose == 0, df['Glucose'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure == 0, df['BloodPressure'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure < 40, df['BloodPressure'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure > 105, df['BloodPressure'].mean())
df['BMI']=df.BMI.mask(df.BMI == 0, df['BMI'].mean())
df['BMI']=df.BMI.mask(df.BMI > 48, df['BMI'].mean())

#Select column based on correlation
df = df.drop(['SkinThickness','Insulin','BloodPressure','DiabetesPedigreeFunction'],axis=1)

#machinelearning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
y = df.iloc[:,-1]
x = df.iloc[:,0:-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
#########################################################################
#FRONTEND

st.title('Prediksi Diagnostik Penyakit Diabetes')

aboutdata=("""
    <p style="text-align: justify;">
    Kumpulan data ini berasal dari <i>National Institute of Diabetes and Digestive and Kidney Diseases</i>. Tujuan dari kumpulan data ini adalah untuk memprediksi secara <b>diagnostik</b> apakah seorang pasien menderita diabetes, berdasarkan pengukuran diagnostik tertentu yang termasuk dalam kumpulan data. Beberapa batasan ditetapkan pada pemilihan instance ini dari database yang lebih besar. Secara khusus, semua pasien di sini adalah perempuan berusia minimal 21 tahun keturunan India. 2 Dari kumpulan data dalam File (.csv) dapat ditemukan beberapa variabel, beberapa di antaranya independent (beberapa variabel prediktor medis) dan hanya satu variabel dependent target (Hasil/<i>Outcome</i>).
    </p>
""")

latarbelakang = ("""
    <p style="text-align: justify;">
    Penyakit diabetes memiliki urgensi yang <b>tinggi</b> di Indonesia. Berdasarkan data dari Kementerian Kesehatan Indonesia pada tahun 2020, prevalensi diabetes di Indonesia terus mengalami peningkatan yang signifikan. Berikut adalah beberapa statistik terkait urgensi penyakit diabetes di Indonesia : 
    
    - Prevalensi Diabetes: Menurut data Riskesdas 2018 (Riset Kesehatan Dasar), prevalensi diabetes di Indonesia pada tahun tersebut adalah sekitar <b>10,7%</b>. Artinya, <b>sekitar 10,7 dari 100 orang di Indonesia memiliki diabetes</b>. 
    
    - Jumlah Penderita Diabetes: Diperkirakan ada sekitar <b>10-11 juta</b> orang di Indonesia yang telah didiagnosis menderita diabetes. Namun, terdapat juga banyak kasus diabetes yang <b>belum terdiagnosis atau belum diketahui</b>.
    </p>
""")

latbel, tendat = st.columns(2)

with latbel:
    st.header('Tentang Data')
    st.write(aboutdata, unsafe_allow_html=True)
with tendat:
    st.header('Latar Belakang')
    st.write(latarbelakang, unsafe_allow_html=True)

st.header('Data Insight')
datafram, corrcol = st.columns(2)
with datafram:
    st.dataframe(df, use_container_width=True)
with corrcol:
    fig, ax = plt.subplots(figsize=(5,2.5))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(fig,use_container_width=True)

diabetespositive = df['Outcome'].loc[df['Outcome'] == 1].count()
percpositive = df['Outcome'].loc[df['Outcome'] == 1].count()/df['Outcome'].count()
diabetesnegative = df['Outcome'].loc[df['Outcome'] == 0].count()
percnegative = df['Outcome'].loc[df['Outcome'] == 0].count()/df['Outcome'].count()

modemode = df.loc[df['Outcome'] == 1]
modeage = modemode['Age'].mode().values[0]

mx_positive, mx_negative, mx_age = st.columns(3)

with mx_positive:
    st.metric(
        "Positif Diabetes (Jumlah)",
        value= diabetespositive
    )
with mx_negative:
    st.metric(
        "Negatif Diabetes (Jumlah)",
        value= diabetesnegative
    )
with mx_age:
    st.metric(
        "Umur dominan penderita diabetes",
        value= f'{modeage} Tahun'
    )

fakta = ("""
    <p style="text-align: justify;">
    <i>Fakta : Berdasarkan heatmap korelasi di atas, Atribut yang paling berpengaruh dalam proses pembuatan proyek ini adalah atribut <b>Glukosa</b> yang artinya <b>Glukosa</b> merupakan faktor utama penyebab Diabetes itu sendiri.</i>
    </p>
""")

st.write(fakta,unsafe_allow_html=True)

st.header('Grafik Penderita Diabetes')

agecount = df.groupby(['Age','Outcome']).size().reset_index(name='count')
agecount['Outcome'] = agecount['Outcome'].replace(0,'Negative').replace(1,'Positive')
custom_colors = alt.Scale(domain=['Negative', 'Positive'], range=['#38B000', '#E63946'])

diabetesbar = alt.Chart(agecount).mark_bar().encode(
        alt.X('Age',title="Age"),
        alt.Y('count',title='Amount'),
        alt.Color('Outcome',scale=custom_colors),
    )

st.altair_chart(diabetesbar, use_container_width=True)

st.header('Grafik Status Diabetes per Umur')

ages = df['Age'].unique()

ageselect = st.selectbox("Umur (berdasarkan data yang ada di database)",sorted(ages))

basedperage = agecount.loc[agecount['Age'] == int(ageselect)]

diabetesbarperage = alt.Chart(basedperage).mark_bar().encode(
        alt.X('Age',title="Age"),
        alt.Y('count',title='Amount'),
        alt.Color('Outcome',scale=custom_colors),
        alt.Column('Outcome', header=alt.Header(title='Outcome'))
    ).properties(
        width=100,
        height=200
    )

st.altair_chart(diabetesbarperage)

tool, insight = st.columns(2)

with tool :
    st.header('Alat Pendeteksi Diabetes (Diagnostik)')

    import numpy as np
    preg = st.number_input('Berapa kali hamil : ', 0,20)
    glu = st.number_input('Level glukosa dalam darah : ',0,300)
    bmi = st.number_input('Indeks Massa Tubuh (IMT) : ',0,50)
    age = st.number_input('Umur : ',0,120)

    negatif = ("""
        <p style="text-align: justify;">
        Anda <b>NEGATIF</b> mengalami penyakit Diabetes
        </p>
    """)
    positif = ("""
        <p style="text-align: justify;">
        Anda <b>POSITIF</b> mengalami penyakit Diabetes
        </p>
    """)

    try:
        pregt = float(preg)
        glut = float(glu)
        bmit = float(bmi)
        aget = int(age)
        demo = np.array([pregt,glut,bmit,aget])
        demo = demo.reshape(1,-1)
        pred = lr.predict(demo)
        pred = pred.reshape(-1,1)
        if st.button('Enter') : 
            if pred[0] == 0:
                st.write(negatif,unsafe_allow_html=True)
            elif pred[0] == 1:
                st.write(positif,unsafe_allow_html=True)
        else:
            st.write('')
    except ValueError:
        st.write("Harap memberikan data secara tepat!")

    from sklearn.metrics import accuracy_score

    st.write("Akurasi Model (Logistic Regression) :\n",round(accuracy_score(y_test,y_pred_lr),2))

    caption = ("""
        <p style="text-align: justify;">
        <i>Penting! : Model ini adalah alat <b>DIAGNOSTIK</b> untuk mendeteksi diabetes saja. Disarankan untuk berkonsultasi ke dokter untuk mendapatkan hasil yang lebih akurat.</i>
        </p>
    """)

    st.caption(caption,unsafe_allow_html=True)
with insight :
    st.header('Nilai Sehat')
    caption2 = ("""
        <p style="text-align: justify;">
        <i>Berikut merupakan nilai normal dari atribut pemicu diabetes yang perlu diperhatikan agar tidak terkena penyakit diabetes.</i>

        <b>NILAI DI BAWAH INI BERDASARKAN DATA YANG ADA PADA DATABASE</b>
        </p>
    """)
    avgglu = round(df['Glucose'].loc[df['Outcome'] == 0].mean(),2)
    avgBMI = round(df['BMI'].loc[df['Outcome'] == 0].mean(),2)
    avgpreg = round(df['Pregnancies'].loc[df['Outcome'] == 0].mean())
    nilainormal = (f"""
        <p style="text-align: justify;">
        Glukosa             : <b>{avgglu}</b>   
        
        Indeks Massa Tubuh  : <b>{avgBMI}</b>  
        
        Jumlah Kehamilan    : <b>{avgpreg} kali</b>  
        </p>
    """)
    st.caption(caption2,unsafe_allow_html=True)
    st.write (nilainormal,unsafe_allow_html=True)
