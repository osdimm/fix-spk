import streamlit as st
import pandas as pd
import numpy as np
import random

st.title("Sistem Pemilihan Vendor - Metode Weighted Product")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Vendor")
    st.dataframe(df)

    vendor_names = df.iloc[:, 0].tolist()
    kriteria = df.columns[1:].tolist()
    data = df.iloc[:, 1:].values

    # Pilih atribut untuk tiap kriteria
    st.subheader("Pilih Jenis Atribut")
    atribut = []
    col = st.columns(len(kriteria))
    for i, k in enumerate(kriteria):
        with col[i]:
            a = st.selectbox(f"{k}", ['cost', 'benefit'], key=f"attr_{i}")
            atribut.append(a)

    # Tombol proses
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

        def generate_population(size, n_kriteria):
            return [np.random.dirichlet(np.ones(n_kriteria)) for _ in range(size)]

        def fitness(weights):
            scores = weighted_product(data, weights, atribut)
            return np.max(scores)

        def crossover(p1, p2):
            alpha = random.random()
            return alpha * p1 + (1 - alpha) * p2

        def mutate(ind, rate=0.1):
            i = random.randint(0, len(ind) - 1)
            ind[i] += np.random.normal(0, rate)
            ind = np.abs(ind)
            return ind / np.sum(ind)

        def evolve(pop, ngen=100):
            for _ in range(ngen):
                pop = sorted(pop, key=lambda x: -fitness(x))
                new_pop = pop[:5]
                while len(new_pop) < len(pop):
                    p1, p2 = random.sample(pop[:10], 2)
                    child = crossover(p1, p2)
                    child = mutate(child)
                    new_pop.append(child)
                pop = new_pop
            return sorted(pop, key=lambda x: -fitness(x))[0]

        populasi = generate_population(size=20, n_kriteria=len(kriteria))
        bobot_terbaik = evolve(populasi, ngen=100)

        st.success("Perhitungan Selesai!")

        st.write("**Bobot Optimal:**", np.round(bobot_terbaik, 3))

        nilai_vendor = weighted_product(data, bobot_terbaik, atribut)

        hasil = pd.DataFrame({
            'Vendor': vendor_names,
            'Skor WP': nilai_vendor
        }).sort_values(by='Skor WP', ascending=False)

        st.subheader("Hasil Perhitungan WP")
        st.dataframe(hasil.reset_index(drop=True))

        st.success(f"✅ Vendor terbaik: **{hasil.iloc[0]['Vendor']}** dengan skor {hasil.iloc[0]['Skor WP']:.4f}")
