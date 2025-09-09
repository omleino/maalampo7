# app.py — Lämmitysvaihtoehtojen vertailu (kevytversio ilman PDF)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------- LASKENTAFUNKTIOT ----------

def laske_kustannukset(investointi, laina_aika, korko, sahkon_hinta, sahkon_kulutus,
                       korjaus_vali, korjaus_hinta, korjaus_laina_aika, sahkon_inflaatio,
                       vuodet: int):
    lyhennys = investointi / laina_aika if laina_aika > 0 else 0
    jaljella = investointi
    hinta = sahkon_hinta
    kustannukset = []
    korjauslainat = []

    for v in range(1, vuodet + 1):
        lyh = lyhennys if v <= laina_aika else 0
        korko_inv = jaljella * (korko / 100) if v <= laina_aika else 0
        if v <= laina_aika:
            jaljella -= lyh
        sahko = hinta * sahkon_kulutus
        if korjaus_vali > 0 and v > 1 and (v - 1) % korjaus_vali == 0:
            korjauslainat.append({
                "jaljella": korjaus_hinta,
                "lyh": (korjaus_hinta / korjaus_laina_aika) if korjaus_laina_aika > 0 else 0,
                "vuosia": korjaus_laina_aika
            })
        korjaus_lyh = korjaus_korot = 0
        for l in korjauslainat:
            if l["vuosia"] > 0:
                korko_l = l["jaljella"] * (korko / 100)
                korjaus_korot += korko_l
                korjaus_lyh += l["lyh"]
                l["jaljella"] -= l["lyh"]
                l["vuosia"] -= 1
        korjauslainat = [l for l in korjauslainat if l["vuosia"] > 0]
        vuosi_kust = lyh + korko_inv + sahko + korjaus_lyh + korjaus_korot
        kustannukset.append(vuosi_kust)
        hinta *= (1 + sahkon_inflaatio / 100)
    return kustannukset

def laske_kaukolampo_kustannukset(kustannus_vuosi0, inflaatio, vuodet: int):
    tulos = []
    h = kustannus_vuosi0
    for _ in range(vuodet):
        tulos.append(h)
        h *= (1 + inflaatio / 100)
    return tulos

def takaisinmaksuaika_investointi(investointi, kaukolampo_sarja, maalampo_sarja):
    vuosittainen_saasto = np.array(kaukolampo_sarja) - np.array(maalampo_sarja)
    kum = np.cumsum(vuosittainen_saasto)
    for vuosi, summa in enumerate(kum, 1):
        if summa >= investointi:
            return vuosi
    return None

def erittely_listat(investointi, laina_aika, korko, sahkon_hinta, kulutus, inflaatio,
                    korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet: int):
    rahoitus, lampo = [], []
    jaljella = investointi
    lyhennys = investointi / laina_aika if laina_aika > 0 else 0
    h = sahkon_hinta
    korjauslainat = []
    for v in range(1, vuodet + 1):
        if v <= laina_aika:
            korko_v = jaljella * (korko / 100)
            rah = lyhennys + korko_v
            jaljella -= lyhennys
        else:
            rah = 0
        if korjaus_vali > 0 and v > 1 and (v - 1) % korjaus_vali == 0:
            korjauslainat.append({
                "jaljella": korjaus_hinta,
                "lyh": (korjaus_hinta / korjaus_laina_aika) if korjaus_laina_aika > 0 else 0,
                "vuosia": korjaus_laina_aika
            })
        korjaus_lyh = korjaus_korot = 0
        for l in korjauslainat:
            if l["vuosia"] > 0:
                korko_l = l["jaljella"] * (korko / 100)
                korjaus_korot += korko_l
                korjaus_lyh += l["lyh"]
                l["jaljella"] -= l["lyh"]
                l["vuosia"] -= 1
        korjauslainat = [l for l in korjauslainat if l["vuosia"] > 0]
        elec = h * kulutus
        lampo.append(elec + korjaus_lyh + korjaus_korot)
        rahoitus.append(rah)
        h *= (1 + inflaatio / 100)
    return rahoitus, lampo

# ---------- SOVELLUS ----------
st.set_page_config(page_title="Lämmitysvaihtoehdot", layout="wide")
st.title("Maalämpö (3 sähkön hintaa) vs Kaukolämpö — kevytversio")

with st.sidebar:
    st.header("Yhteiset oletukset")
    investointi = st.number_input("Investointi (€)", min_value=0.0, value=650000.0, step=10000.0)
    laina_aika = st.slider("Laina-aika (v)", 5, 40, 20)
    korko = st.number_input("Korko (%/v)", min_value=0.0, value=3.0, step=0.1)
    kulutus = st.number_input("Sähkönkulutus (kWh/v)", min_value=0.0, value=180000.0, step=10000.0)
    inflaatio = st.number_input("Sähkön inflaatio (%/v)", min_value=0.0, value=2.0, step=0.1)
    korjaus_vali = st.slider("Korjausväli (v)", 5, 30, 15)
    korjaus_hinta = st.number_input("Korjauksen hinta (€)", min_value=0.0, value=20000.0, step=5000.0)
    korjaus_laina_aika = st.slider("Korjauslaina (v)", 1, 30, 10)
    maalampo_kk_kulu = st.number_input("Maalämmön kuukausikustannus (€ / kk)", min_value=0.0, value=100.0, step=10.0)
    st.header("Sähkön hinnat")
    h1 = st.number_input("Vaihtoehto A (€/kWh)", min_value=0.0, value=0.08, step=0.01)
    h2 = st.number_input("Vaihtoehto B (€/kWh)", min_value=0.0, value=0.12, step=0.01)
    h3 = st.number_input("Vaihtoehto C (€/kWh)", min_value=0.0, value=0.16, step=0.01)
    st.header("Kaukolämpö")
    kl0 = st.number_input("Kaukolämpö/vuosi (€)", min_value=0.0, value=85000.0, step=5000.0)
    kl_inf = st.number_input("Kaukolämmön inflaatio (%/v)", min_value=0.0, value=2.0, step=0.1)
    st.header("Maksuperuste")
    neliot = st.number_input("Maksavat neliöt (m²)", min_value=1.0, value=1000.0, step=100.0)

# Aikajänne
vuodet = int(laina_aika) + 1
vuosilista = list(range(1, vuodet + 1))

# Laskelmat
ml_extra = maalampo_kk_kulu * 12
kl = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)
ml1 = [v + ml_extra for v in laske_kustannukset(investointi, laina_aika, korko, h1, kulutus,
                                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)]
ml2 = [v + ml_extra for v in laske_kustannukset(investointi, laina_aika, korko, h2, kulutus,
                                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)]
ml3 = [v + ml_extra for v in laske_kustannukset(investointi, laina_aika, korko, h3, kulutus,
                                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)]

# Kuvaaja
fig, ax = plt.subplots()
ax.plot(vuosilista, kl, "--", label="Kaukolämpö")
ax.plot(vuosilista, ml1, label=f"Maalämpö A ({h1:.2f} €/kWh)")
ax.plot(vuosilista, ml2, label=f"Maalämpö B ({h2:.2f} €/kWh)")
ax.plot(vuosilista, ml3, label=f"Maalämpö C ({h3:.2f} €/kWh)")
ax.set_xlabel("Vuosi"); ax.set_ylabel("Kustannus (€)")
ax.set_title("Lämmityskustannukset")
ax.grid(True); ax.legend()
st.pyplot(fig, use_container_width=True)

# Taulukko vuosittain vain vaihtoehto B
rahoitus, lampo_b = erittely_listat(investointi, laina_aika, korko, h2, kulutus, inflaatio,
                                    korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
kl_vastike = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)

vuosittainen_df = pd.DataFrame({
    "Vuosi": vuosilista,
    "ML B — ilman rahoitusta €/m²/kk": [(lampo_b[y-1] + ml_extra) / neliot / 12 for y in vuosilista],
    "ML B — yhteensä €/m²/kk": [(rahoitus[y-1] + lampo_b[y-1] + ml_extra) / neliot / 12 for y in vuosilista],
    "Kaukolämpö €/m²/kk": [kl_vastike[y-1] / neliot / 12 for y in vuosilista],
}).set_index("Vuosi")

st.markdown("### Vastikkeet vuoden välein (€/m²/kk) — Vaihtoehto B")
st.dataframe(vuosittainen_df.style.format("{:.2f}"), use_container_width=True)

# Vuosittainen säästö B vs KL per asunto
asunnot = [50, 74, 86]
vuosittainen_saasto_euro_talo = [kl[y-1] - ml2[y-1] for y in vuosilista]
saasto_df = pd.DataFrame(index=vuosilista)
for neliot_asunto in asunnot:
    saasto_df[f"{neliot_asunto} m²"] = [
        (vuosittainen_saasto_euro_talo[y-1] * (neliot_asunto / neliot)) for y in vuosilista
    ]
saasto_df.index.name = "Vuosi"
st.markdown("### Vuosittainen säästö (Maalämpö B vs. Kaukolämpö) €/vuosi per asunto")
st.dataframe(saasto_df.style.format("{:.2f}"), use_container_width=True)
