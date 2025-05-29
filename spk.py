import streamlit as st
import pandas as pd
import numpy as np
import random

st.title("Sistem Pemilihan Vendor - Metode Weighted Product dengan GA & Manual Bobot")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Vendor / Alternatif")
    st.dataframe(df)

    vendor_names = df.iloc[:, 0].tolist()
    kriteria = df.columns[1:].tolist()
    data = df.iloc[:, 1:].values

    st.subheader("Pilih Jenis Atribut untuk Setiap Kriteria")
    atribut = []
    cols = st.columns(len(kriteria))
    for i, k in enumerate(kriteria):
        with cols[i]:
            a = st.selectbox(f"{k}", ['cost', 'benefit'], key=f"attr_{i}")
            atribut.append(a)

    st.subheader("Pilih Metode Pembobotan")
    metode = st.radio("Metode pembobotan yang ingin digunakan:", ("Manual Input Bobot", "Auto Optimasi Bobot (GA)"))

    if metode == "Manual Input Bobot":
        st.write("Masukkan bobot untuk tiap kriteria (total tidak harus 1, akan dinormalisasi otomatis):")
        bobot_manual = []
        cols_bobot = st.columns(len(kriteria))
        for i, k in enumerate(kriteria):
            with cols_bobot[i]:
                b = st.number_input(f"Bobot {k}", min_value=0.0, max_value=100.0, value=1.0, step=0.1, format="%.3f", key=f"manual_bobot_{i}")
                bobot_manual.append(b)
        bobot_manual = np.array(bobot_manual)
        if np.sum(bobot_manual) == 0:
            st.warning("Total bobot manual adalah 0, harap masukkan bobot yang valid.")
        else:
            bobot_manual = bobot_manual / np.sum(bobot_manual)

    if st.button("Hitung Vendor Terbaik"):

        def weighted_product(matrix, weights, atribut):
            weights = weights / np.sum(weights)
            norm = np.zeros_like(matrix, dtype=float)
            for j in range(matrix.shape[1]):
                if atribut[j] == 'benefit':
                    norm[:, j] = matrix[:, j] / np.max(matrix[:, j])
                else:
                    norm[:, j] = np.min(matrix[:, j]) / matrix[:, j]
            wp_score = np.prod(norm ** weights, axis=1)
            return wp_score

        if metode == "Auto Optimasi Bobot (GA)":
            # GA functions
            def generate_population(size, n_kriteria):
                return [np.random.dirichlet(np.ones(n_kriteria)) for _ in range(size)]

            def fitness(weights):
                scores = weighted_product(data, weights, atribut)
                return np.sum(scores)

            def crossover(p1, p2):
                alpha = random.random()
                return alpha * p1 + (1 - alpha) * p2

            def mutate(ind, rate=0.05, min_weight=0.05):
                i = random.randint(0, len(ind) - 1)
                ind[i] += np.random.normal(0, rate)
                ind = np.abs(ind)
                ind = np.clip(ind, min_weight, None)
                return ind / np.sum(ind)

            def evolve(pop, ngen=300):
                for _ in range(ngen):
                    pop = sorted(pop, key=lambda x: -fitness(x))
                    new_pop = pop[:5]  # elitisme
                    while len(new_pop) < len(pop):
                        p1, p2 = random.sample(pop[:10], 2)
                        child = crossover(p1, p2)
                        child = mutate(child)
                        new_pop.append(child)
                    pop = new_pop
                return sorted(pop, key=lambda x: -fitness(x))[0]

            populasi = generate_population(size=50, n_kriteria=len(kriteria))
            bobot = evolve(populasi, ngen=300)
            st.success("Pembobotan otomatis selesai!")
        else:
            if np.sum(bobot_manual) == 0:
                st.warning("Masukkan bobot manual dulu yang valid sebelum menghitung.")
                st.stop()
            bobot = bobot_manual
            st.success("Pembobotan manual digunakan.")

        st.write("**Bobot yang digunakan:**", np.round(bobot, 3))

        nilai_vendor = weighted_product(data, bobot, atribut)
        hasil = pd.DataFrame({
            'Vendor': vendor_names,
            'Skor WP': nilai_vendor
        }).sort_values(by='Skor WP', ascending=False)

        st.subheader("Hasil Perhitungan WP")
        st.dataframe(hasil.reset_index(drop=True))

        st.success(f"✅ Vendor terbaik: **{hasil.iloc[0]['Vendor']}** dengan skor {hasil.iloc[0]['Skor WP']:.4f}")
