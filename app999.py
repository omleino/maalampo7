# app.py — Lämmitysvaihtoehtojen vertailu (sisältää PDF-raportin)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile

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

def _format_df_for_pdf(df: pd.DataFrame) -> list:
    df_reset = df.reset_index()
    header = df_reset.columns.to_list()
    body_raw = df_reset.values.tolist()
    body_fmt = []
    for row in body_raw:
        new_row = []
        for v in row:
            if isinstance(v, (int, float, np.integer, np.floating)):
                new_row.append(f"{v:.2f}")
            else:
                new_row.append(v)
        body_fmt.append(new_row)
    return [header] + body_fmt

def luo_pdf(kaavio, vuosittainen_df, saasto_df, lainaosuus, vuodet):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Lämmitysvaihtoehtojen vertailuraportti", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))
    # Kuva
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    kaavio.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
    elements.append(Image(tmpfile.name, width=6*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))
    # Taulukko 1
    elements.append(Paragraph("Vastikkeet vuoden välein (€/m²/kk) — Vaihtoehto B", styles['Heading2']))
    table_data = _format_df_for_pdf(vuosittainen_df)
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),
                               ('GRID',(0,0),(-1,-1),0.25,colors.black),
                               ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                               ('FONTSIZE',(0,0),(-1,-1),7),
                               ('ALIGN',(1,1),(-1,-1),'RIGHT')]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))
    # Taulukko 2
    elements.append(Paragraph("Vuosittainen säästö (Maalämpö B vs. Kaukolämpö) €/vuosi per asunto", styles['Heading2']))
    table_data_s = _format_df_for_pdf(saasto_df)
    table_s = Table(table_data_s, repeatRows=1)
    table_s.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),
                                 ('GRID',(0,0),(-1,-1),0.25,colors.black),
                                 ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                                 ('FONTSIZE',(0,0),(-1,-1),7),
                                 ('ALIGN',(1,1),(-1,-1),'RIGHT')]))
    elements.append(table_s)
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Lainaosuus: {lainaosuus:,.0f} €/m²", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- SOVELLUS ----------
st.set_page_config(page_title="Lämmitysvaihtoehdot", layout="wide")
st.title("Maalämpö (3 sähkön hintaa) vs Kaukolämpö")

with st.sidebar:
    investointi = st.number_input("Investointi (€)", 0.0, value=650000.0, step=10000.0)
    laina_aika = st.slider("Laina-aika (v)", 5, 40, 20)
    korko = st.number_input("Korko (%/v)", 0.0, value=3.0, step=0.1)
    kulutus = st.number_input("Sähkönkulutus (kWh/v)", 0.0, value=180000.0, step=10000.0)
    inflaatio = st.number_input("Sähkön inflaatio (%/v)", 0.0, value=2.0, step=0.1)
    korjaus_vali = st.slider("Korjausväli (v)", 5, 30, 15)
    korjaus_hinta = st.number_input("Korjauksen hinta (€)", 0.0, value=20000.0, step=5000.0)
    korjaus_laina_aika = st.slider("Korjauslaina (v)", 1, 30, 10)
    maalampo_kk_kulu = st.number_input("Maalämmön kuukausikustannus (€ / kk)", 0.0, value=100.0, step=10.0)
    h2 = st.number_input("Sähkön hinta B (€/kWh)", 0.0, value=0.12, step=0.01)
    kl0 = st.number_input("Kaukolämpö/vuosi (€)", 0.0, value=85000.0, step=5000.0)
    kl_inf = st.number_input("Kaukolämmön inflaatio (%/v)", 0.0, value=2.0, step=0.1)
    neliot = st.number_input("Maksavat neliöt (m²)", 1.0, value=1000.0, step=100.0)

vuodet = int(laina_aika) + 1
vuosilista = list(range(1, vuodet + 1))

ml_extra = maalampo_kk_kulu * 12
kl = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)
ml2 = [v + ml_extra for v in laske_kustannukset(investointi, laina_aika, korko, h2, kulutus,
                                                korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)]

# Kuvaaja
fig, ax = plt.subplots()
ax.plot(vuosilista, kl, "--", label="Kaukolämpö")
ax.plot(vuosilista, ml2, label=f"Maalämpö B ({h2:.2f} €/kWh)")
ax.set_xlabel("Vuosi"); ax.set_ylabel("Kustannus (€)")
ax.set_title("Lämmityskustannukset")
ax.grid(True); ax.legend()
st.pyplot(fig, use_container_width=True)

# Taulukot
rahoitus, lampo_b = erittely_listat(investointi, laina_aika, korko, h2, kulutus, inflaatio,
                                    korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
kl_vastike = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)
vuosittainen_df = pd.DataFrame({
    "Vuosi": vuosilista,
    "ML B — ilman rahoitusta €/m²/kk": [(lampo_b[y-1] + ml_extra)/neliot/12 for y in vuosilista],
    "ML B — yhteensä €/m²/kk": [(rahoitus[y-1]+lampo_b[y-1]+ml_extra)/neliot/12 for y in vuosilista],
    "Kaukolämpö €/m²/kk": [kl_vastike[y-1]/neliot/12 for y in vuosilista],
}).set_index("Vuosi")
st.dataframe(vuosittainen_df.style.format("{:.2f}"), use_container_width=True)

vuosittainen_saasto_euro_talo = [kl[y-1] - ml2[y-1] for y in vuosilista]
asunnot = [50,74,86]
saasto_df = pd.DataFrame(index=vuosilista)
for neliot_asunto in asunnot:
    saasto_df[f"{neliot_asunto} m²"] = [
        (vuosittainen_saasto_euro_talo[y-1]*(neliot_asunto/neliot)) for y in vuosilista
    ]
saasto_df.index.name="Vuosi"
st.dataframe(saasto_df.style.format("{:.2f}"), use_container_width=True)

# PDF-lataus
lainaosuus = investointi/neliot if neliot>0 else 0
try:
    pdf = luo_pdf(fig, vuosittainen_df, saasto_df, lainaosuus, vuodet)
    st.download_button("📄 Lataa PDF-raportti", data=pdf, file_name="raportti.pdf", mime="application/pdf")
except Exception as e:
    st.error(f"PDF:n luonnissa virhe: {e}")
