import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@st.cache_data
def baca_hasil(folder):
    """Baca file hasil notebook dari folder hasil_listrik."""
    nama_csv = {
        "daily": "daily.csv",
        "test": "evaluasi_test.csv",
        "prediksi": "prediksi_test.csv",
        "nodes": "graph_nodes.csv",
        "edges": "graph_edges.csv",
        "threshold": "graph_threshold.csv",
    }
    kurang = [nama for nama in [*nama_csv.values(), "run_info.json"]
              if not (folder / nama).is_file()]
    if kurang:
        raise FileNotFoundError("File hasil belum lengkap: " + ", ".join(kurang))
    hasil = {kunci: pd.read_csv(folder / nama)
             for kunci, nama in nama_csv.items()}
    hasil["info"] = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    return hasil


def gambar_graph(nodes, edges):
    """Posisi jam dibuat tetap supaya hubungan antar jam mudah dibaca."""
    fig, ax = plt.subplots(figsize=(7, 7))
    posisi = {}
    for jam in range(24):
        sudut = math.pi / 2 - 2 * math.pi * jam / 24
        posisi[jam] = (math.cos(sudut), math.sin(sudut))

    for sisi in edges.itertuples(index=False):
        x1, y1 = posisi[int(sisi.hour_a)]
        x2, y2 = posisi[int(sisi.hour_b)]
        ax.plot([x1, x2], [y1, y2], color="#64748b", alpha=0.65,
                linewidth=1 + 3 * float(sisi.correlation), zorder=1)

    warna = plt.get_cmap("tab20")
    for titik in nodes.itertuples(index=False):
        jam = int(titik.hour)
        x, y = posisi[jam]
        ax.scatter(x, y, s=230 + 160 * float(titik.mean_hourly_kwh),
                   color=warna((int(titik.community) - 1) % 20),
                   edgecolor="white", linewidth=1.5, zorder=2)
        ax.text(x * 1.15, y * 1.15, f"{jam:02d}", ha="center", va="center",
                fontsize=9)

    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Analisis konsumsi listrik", layout="wide")
st.title("Analisis konsumsi listrik rumah tangga")
st.caption("Data UCI Individual Household Electric Power Consumption; hasil dari notebook analisis.")

folder_hasil = Path(__file__).resolve().parent / "hasil_listrik"

try:
    data = baca_hasil(folder_hasil)
except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as err:
    st.error(f"Hasil notebook tidak dapat dibaca: {err}")
    st.stop()

daily = data["daily"]
daily["day"] = pd.to_datetime(daily["day"])
lengkap = daily.dropna(subset=["energy_kwh"]).copy()
test = data["test"]
prediksi = data["prediksi"]
prediksi["day"] = pd.to_datetime(prediksi["day"])
nodes = data["nodes"]
edges = data["edges"]
info = data["info"]
model = info["chosen_model"]

if model not in prediksi.columns or model not in test["model"].values:
    st.error("Nama model utama tidak cocok dengan tabel hasil.")
    st.stop()

mae_model = float(test.loc[test["model"] == model, "MAE_kwh"].iloc[0])
mae_baseline = float(test.loc[test["model"] == "Baseline lag 1", "MAE_kwh"].iloc[0])
perbaikan = (mae_baseline - mae_model) / mae_baseline * 100

st.subheader("1. Konsumsi harian dan kelengkapan data")
c1, c2, c3 = st.columns(3)
c1.metric("Hari penuh", f"{len(lengkap):,}")
c2.metric("Hari tidak penuh", f"{len(daily) - len(lengkap):,}")
c3.metric("Rata-rata hari penuh", f"{lengkap['energy_kwh'].mean():.1f} kWh")
st.line_chart(lengkap.set_index("day")["energy_kwh"], y_label="kWh/hari")
st.info("**Makna bisnis:** kebutuhan energi berubah dari hari ke hari. Gunakan riwayat hari penuh untuk merencanakan anggaran dan memantau hari dengan konsumsi tinggi. Hari tidak penuh tidak dipakai sebagai target model.")

st.subheader("2. Prediksi kebutuhan listrik besok (Machine Learning)")
c1, c2, c3 = st.columns(3)
c1.metric("MAE model utama", f"{mae_model:.2f} kWh/hari")
c2.metric("MAE tebak seperti kemarin", f"{mae_baseline:.2f} kWh/hari")
c3.metric("Penurunan kesalahan", f"{perbaikan:.1f}%")
kiri, kanan = st.columns([1, 1.3])
with kiri:
    st.caption("MAE pada data test; semakin kecil semakin baik")
    st.bar_chart(test.set_index("model")["MAE_kwh"], horizontal=True,
                 x_label="kWh/hari")
with kanan:
    st.caption("Konsumsi aktual dan prediksi model utama pada data test")
    garis = prediksi.set_index("day")[["actual_kwh", model]].rename(
        columns={"actual_kwh": "Aktual", model: "Prediksi"})
    st.line_chart(garis, y_label="kWh/hari")
st.info(f"**Makna bisnis:** {model} mengurangi rata-rata selisih prediksi sebesar "
        f"{perbaikan:.1f}% dibanding perkiraan memakai konsumsi kemarin. "
        "Prediksi membantu menyiapkan anggaran atau perkiraan beban esok hari, "
        "tetapi lonjakan mendadak masih bisa meleset.")
st.caption("Model dipilih dari validation, lalu dinilai sekali pada test; hasil berasal dari satu rumah.")

st.subheader("3. Hubungan pola antar jam (Graph Analytics)")
jam_puncak = nodes.loc[nodes["mean_hourly_kwh"].idxmax()]
sisi_terkuat = edges.loc[edges["correlation"].idxmax()]
g1, g2, g3 = st.columns(3)
g1.metric("Jam dengan energi rata-rata tertinggi", f"{int(jam_puncak['hour']):02d}:00")
g2.metric("Hubungan jam terkuat", f"{int(sisi_terkuat['hour_a']):02d}–{int(sisi_terkuat['hour_b']):02d}")
g3.metric("Korelasi hubungan terkuat", f"{sisi_terkuat['correlation']:.2f}")
kiri, kanan = st.columns([1, 1])
with kiri:
    st.pyplot(gambar_graph(nodes, edges), clear_figure=True)
    st.caption("Titik = jam; ukuran = rata-rata kWh/jam; warna = komunitas; garis = korelasi positif ≥ 0,70.")
with kanan:
    st.caption("Rata-rata energi tiap jam pada hari penuh periode train")
    rata_jam = nodes.sort_values("hour").set_index("hour")["mean_hourly_kwh"]
    st.bar_chart(rata_jam, x_label="Jam", y_label="kWh/jam")
    st.caption("Kepekaan jumlah hubungan terhadap ambang korelasi")
    st.dataframe(data["threshold"], hide_index=True, width="stretch")
st.info(f"**Makna bisnis:** jam {int(jam_puncak['hour']):02d}:00 memiliki "
        "rata-rata energi tertinggi pada data train; ini waktu yang relevan untuk "
        "memeriksa beban. Jam yang terhubung menunjukkan pola naik-turun yang "
        "serupa antar hari dan dapat dipantau sebagai kelompok. Korelasi bukan "
        "bukti bahwa satu jam menyebabkan konsumsi pada jam lain.")
st.caption("Graph dibangun dari hari penuh pada periode train saja. Jaringan berubah saat ambang korelasi diubah; temuan ini tidak otomatis berlaku untuk rumah lain.")
