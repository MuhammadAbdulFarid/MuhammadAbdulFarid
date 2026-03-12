import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETTING TAMPILAN WEB ---
st.set_page_config(page_title="Zakat-AI", page_icon="🤲")
st.title("🤖 Zakat-AI: Prediksi Kelayakan Mustahiq")
st.write("Masukkan data warga di bawah ini, dan biarkan AI menentukan kelayakannya secara objektif.")
st.markdown("---")

# --- 2. AI BELAJAR DI BELAKANG LAYAR ---
# Load data
df = pd.read_csv('dataset_zakat.csv')

# Ubah teks jadi angka
df['Kondisi_Rumah'] = df['Kondisi_Rumah'].map({'Jelek': 0, 'Sedang': 1, 'Bagus': 2})

# Siapin Soal dan Jawaban
X = df.drop('Target_Kelayakan', axis=1)
y = df['Target_Kelayakan']

# Latih Model Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X, y)

# --- 3. BIKIN FORM INPUT BUAT USER ---
col1, col2 = st.columns(2) # Bikin jadi 2 kolom biar rapi

with col1:
    umur = st.number_input("Berapa Umurnya?", min_value=18, max_value=100, value=30)
    anak = st.number_input("Jumlah Tanggungan Anak?", min_value=0, max_value=15, value=1)

with col2:
    pendapatan = st.number_input("Gaji/Pendapatan Perbulan (Rp)", min_value=0, value=2500000)
    rumah = st.selectbox("Bagaimana Kondisi Rumahnya?", ['Jelek', 'Sedang', 'Bagus'])

st.markdown("---")

# --- 4. TOMBOL PREDIKSI ---
if st.button("🔍 Cek Kelayakan Sekarang"):
    
    # Terjemahin inputan rumah jadi angka buat AI
    mapping_rumah = {'Jelek': 0, 'Sedang': 1, 'Bagus': 2}
    rumah_angka = mapping_rumah[rumah]
    
    # Bungkus data barunya
    data_baru = pd.DataFrame({
        'Umur': [umur],
        'Pendapatan_Perbulan': [pendapatan],
        'Jumlah_Anak': [anak],
        'Kondisi_Rumah': [rumah_angka]
    })
    
    # AI Nebak!
    hasil_prediksi = model_rf.predict(data_baru)
    
    # Tunjukin Hasilnya ke Layar
    if hasil_prediksi[0] == 1:
        st.success("✅ HASIL AI: Warga ini **LAYAK** mendapatkan bantuan zakat.")
        st.balloons() # Munculin animasi balon meledak biar juri kaget wkwk
    else:
        st.error("❌ HASIL AI: Warga ini **TIDAK LAYAK** mendapatkan bantuan zakat.")