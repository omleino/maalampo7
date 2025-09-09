# app.py — Lämmitysvaihtoehtojen vertailu (horisontti = laina-aika + 1) + PDF-raportti
# - PDF: vain vuosittainen vastiketaulukko + vuosittainen säästötaulukko (B vs KL)
# - PDF: kaikki numerot 2 desimaalin tarkkuudella
# - Ei takaisinmaksuaikaa PDF:ssä (näkyy vain UI:ssa)

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

# ---------- LASKENTAFUNKTIOT (dynaaminen vuosihorisontti) ----------

def laske_kustannukset(investointi, laina_aika, korko, sahkon_hinta, sahkon_kulutus,
                       korjaus_vali, korjaus_hinta, korjaus_laina_aika, sahkon_inflaatio,
                       vuodet: int):
    """Palauttaa listan vuosittaisista kokonaiskustannuksista pituudella 'vuodet'."""
    lyhennys = investointi / laina_aika if laina_aika > 0 else 0
    jaljella = investointi
    hinta = sahkon_hinta
    kustannukset = []
    korjauslainat = []

    for v in range(1, vuodet + 1):
        # Pääinvestoinnin lyhennys ja korko
        lyh = lyhennys if v <= laina_aika else 0
        korko_inv = jaljella * (korko / 100) if v <= laina_aika else 0
        if v <= laina_aika:
            jaljella -= lyh

        # Sähkökulu
        sahko = hinta * sahkon_kulutus

        # Korjausinvestointien lainat (toistuvat tietyin välein)
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

        # Sähkön hinta nousee inflaatiolla
        hinta *= (1 + sahkon_inflaatio / 100)

    return kustannukset


def laske_kaukolampo_kustannukset(kustannus_vuosi0, inflaatio, vuodet: int):
    """Kaukolämmön vuosikustannukset listana pituudella 'vuodet'."""
    tulos = []
    h = kustannus_vuosi0
    for _ in range(vuodet):
        tulos.append(h)
        h *= (1 + inflaatio / 100)
    return tulos


def takaisinmaksuaika_investointi(investointi, kaukolampo_sarja, maalampo_sarja):
    """Näkyy UI:ssa, ei PDF:ssä."""
    vuosittainen_saasto = np.array(kaukolampo_sarja) - np.array(maalampo_sarja)
    kum = np.cumsum(vuosittainen_saasto)
    for vuosi, summa in enumerate(kum, 1):
        if summa >= investointi:
            return vuosi
    return None


def erittely_listat(investointi, laina_aika, korko, sahkon_hinta, kulutus, inflaatio,
                    korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet: int):
    """Palauttaa (rahoitus, lämmitys) -listat pituudella 'vuodet'."""
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
    """PDF-taulukon data: otsikkorivi + rivit, numerot 2 desimaalina."""
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


def luo_pdf(kaavio, lainaosuus, syotteet, vuodet_teksti):
    """PDF: kuva + vuosittainen vastiketaulukko + vuosittainen säästötaulukko."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Lämmitysvaihtoehtojen vertailuraportti", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Syötetyt arvot:", styles['Heading2']))
    for nimi, arvo in syotteet.items():
        if isinstance(arvo, pd.DataFrame):
            continue
        elements.append(Paragraph(f"{nimi}: {arvo}", styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    kaavio.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
    elements.append(Paragraph(f"Lämmityskustannukset {vuodet_teksti} ajalta", styles['Heading2']))
    elements.append(Image(tmpfile.name, width=6*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))

    if "vuosittainen_df" in syotteet and isinstance(syotteet["vuosittainen_df"], pd.DataFrame):
        vdf = syotteet["vuosittainen_df"]
        cols = [
            "ML A — ilman rahoitusta €/m²/kk",
            "ML B — ilman rahoitusta €/m²/kk",
            "ML C — ilman rahoitusta €/m²/kk",
            "ML A — yhteensä €/m²/kk",
            "ML B — yhteensä €/m²/kk",
            "ML C — yhteensä €/m²/kk",
            "Kaukolämpö €/m²/kk",
        ]
        cols = [c for c in cols if c in vdf.columns]
        vdf_ordered = vdf[cols]
        vdf_ordered.index.name = "Vuosi"

        elements.append(Paragraph("Vastikkeet vuoden välein (€/m²/kk)", styles['Heading2']))
        table_data = _format_df_for_pdf(vdf_ordered)
        table_v = Table(table_data, repeatRows=1)
        table_v.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        elements.append(table_v)
        elements.append(Spacer(1, 0.3 * inch))

    if "saasto_df" in syotteet and isinstance(syotteet["saasto_df"], pd.DataFrame):
        sdf = syotteet["saasto_df"]
        cols_s = [c for c in ["50 m²", "74 m²", "86 m²"] if c in sdf.columns]
        sdf_ordered = sdf[cols_s]
        sdf_ordered.index.name = "Vuosi"

        elements.append(Paragraph("Vuosittainen säästö (Maalämpö B vs. Kaukolämpö) €/vuosi per asunto", styles['Heading2']))
        table_data_s = _format_df_for_pdf(sdf_ordered)
        table_s = Table(table_data_s, repeatRows=1)
        table_s.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
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

# Dynaaminen aikajänne: laina-aika + 1 vuosi
vuodet = int(laina_aika) + 1
vuosilista = list(range(1, vuodet + 1))
vuodet_teksti = f"{vuodet} vuoden"

# Laskelmat
ml_extra = maalampo_kk_kulu * 12
kl = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)
ml1_pohja = laske_kustannukset(investointi, laina_aika, korko, h1, kulutus,
                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)
ml2_pohja = laske_kustannukset(investointi, laina_aika, korko, h2, kulutus,
                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)
ml3_pohja = laske_kustannukset(investointi, laina_aika, korko, h3, kulutus,
                               korjaus_vali, korjaus_hinta, korjaus_laina_aika, inflaatio, vuodet)

ml1 = [v + ml_extra for v in ml1_pohja]
ml2 = [v + ml_extra for v in ml2_pohja]
ml3 = [v + ml_extra for v in ml3_pohja]

# Otsikko ja info
st.subheader(f"Aikajänne: {vuodet} vuotta (laina-aika {laina_aika} v + 1 v)")
st.caption("Kaikki sarjat ja taulukot lasketaan dynaamisesti valitun aikajänteen mukaan.")

# Kaavio
fig, ax = plt.subplots()
ax.plot(vuosilista, kl, "--", label="Kaukolämpö")
ax.plot(vuosilista, ml1, label=f"Maalämpö A ({h1:.2f} €/kWh)")
ax.plot(vuosilista, ml2, label=f"Maalämpö B ({h2:.2f} €/kWh)")
ax.plot(vuosilista, ml3, label=f"Maalämpö C ({h3:.2f} €/kWh)")
ax.set_xlabel("Vuosi"); ax.set_ylabel("Kustannus (€)")
ax.set_title(f"Lämmityskustannukset {vuodet_teksti} ajalta")
ax.grid(True); ax.legend()
ax.set_xlim(1, vuodet)
st.pyplot(fig, use_container_width=True)

# --- Taulukko 5 v välein (näytetään UI:ssa edelleen) ---
yrs5 = list(range(5, vuodet + 1, 5))
if len(yrs5) == 0:
    yrs5 = [vuodet]

rahoitus, _ = erittely_listat(investointi, laina_aika, korko, h1, kulutus, inflaatio,
                              korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
_, lampo1 = erittely_listat(investointi, laina_aika, korko, h1, kulutus, inflaatio,
                            korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
_, lampo2 = erittely_listat(investointi, laina_aika, korko, h2, kulutus, inflaatio,
                            korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
_, lampo3 = erittely_listat(investointi, laina_aika, korko, h3, kulutus, inflaatio,
                            korjaus_vali, korjaus_hinta, korjaus_laina_aika, vuodet)
kl_vastike = laske_kaukolampo_kustannukset(kl0, kl_inf, vuodet)

tbl = pd.DataFrame({
    "Vuosi": yrs5,
    "Rahoitusvastike €/m²/kk": [rahoitus[y-1]/neliot/12 for y in yrs5],
    "Lämmitysvastike A €/m²/kk": [(lampo1[y-1] + ml_extra)/neliot/12 for y in yrs5],
    "Lämmitysvastike B €/m²/kk": [(lampo2[y-1] + ml_extra)/neliot/12 for y in yrs5],
    "Lämmitysvastike C €/m²/kk": [(lampo3[y-1] + ml_extra)/neliot/12 for y in yrs5],
    "Kaukolämpö €/m²/kk": [kl_vastike[y-1]/neliot/12 for y in yrs5]
}).set_index("Vuosi")

st.markdown("### Rahoitus- ja lämmitysvastikkeet €/m²/kk (5 v välein)")
st.dataframe(tbl.style.format("{:.2f}"), use_container_width=True)

# --- UUSI TAULUKKO: Vastikkeet vuoden välein (€/m²/kk) ---
# Järjestys: ensin ilman investointia (A, B, C), sitten investoinnin kanssa (A, B, C), lopuksi kaukolämpö
vuodet_range = list(range(1, vuodet + 1))
vuosittainen_df = pd.DataFrame({
    "Vuosi": vuodet_range,
    "ML A — ilman rahoitusta €/m²/kk": [ (lampo1[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "ML B — ilman rahoitusta €/m²/kk": [ (lampo2[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "ML C — ilman rahoitusta €/m²/kk": [ (lampo3[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "ML A — yhteensä €/m²/kk": [ (rahoitus[y-1] + lampo1[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "ML B — yhteensä €/m²/kk": [ (rahoitus[y-1] + lampo2[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "ML C — yhteensä €/m²/kk": [ (rahoitus[y-1] + lampo3[y-1] + ml_extra) / neliot / 12 for y in vuodet_range ],
    "Kaukolämpö €/m²/kk": [ kl_vastike[y-1] / neliot / 12 for y in vuodet_range ],
}).set_index("Vuosi")

st.markdown("### Vastikkeet vuoden välein (€/m²/kk) — ilman investointia / yhteensä / kaukolämpö")
st.dataframe(vuosittainen_df.style.format("{:.2f}"), use_container_width=True)

# CSV-lataus vuosittaisesta taulukosta
csv_vuosittainen = vuosittainen_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Lataa vuosittainen vastiketaulukko (CSV)",
                   data=csv_vuosittainen,
                   file_name="vastikkeet_vuosittain.csv",
                   mime="text/csv")

# --- VUOSITTAINEN SÄÄSTÖ VUODESSA: Maalämpö B vs Kaukolämpö (per asunto) ---
asunnot = [50, 74, 86]
saasto_vuodet = list(range(1, vuodet + 1))

# Vuotuinen säästö koko talolle (Kaukolämpö - Maalämpö B)
vuosittainen_saasto_euro = [kl[y-1] - ml2[y-1] for y in saasto_vuodet]

# DataFrame: rivit = vuodet, sarakkeet = asunnon koko (€/vuosi per asunto)
saasto_df = pd.DataFrame(index=saasto_vuodet)
for neliot_asunto in asunnot:
    saasto_df[f"{neliot_asunto} m²"] = [
        vuosittainen_saasto_euro[y-1] / neliot_asunto for y in saasto_vuodet
    ]
saasto_df.index.name = "Vuosi"

st.markdown("### Vuosittainen säästö (Maalämpö B vs. Kaukolämpö) €/vuosi per asunto")
st.dataframe(saasto_df.style.format("{:.2f}"), use_container_width=True)

# CSV-lataus säästötaulukosta
csv_saasto = saasto_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Lataa säästötaulukko (CSV)",
                   data=csv_saasto,
                   file_name="saastot_asunnoittain.csv",
                   mime="text/csv")

# Takaisinmaksuaika (vain UI:ssa, ei PDF:ssä)
pb1 = takaisinmaksuaika_investointi(investointi, kl, ml1)
pb2 = takaisinmaksuaika_investointi(investointi, kl, ml2)
pb3 = takaisinmaksuaika_investointi(investointi, kl, ml3)

f_txt = lambda v: f"{int(v)} vuotta" if v and float(v).is_integer() else (f"{v:.1f} vuotta" if v else f"ei {vuodet_teksti} ajalla")
st.markdown("### Investoinnin takaisinmaksuaika")
st.write(f"**Maalämpö A ({h1:.2f} €/kWh):** {f_txt(pb1)}")
st.write(f"**Maalämpö B ({h2:.2f} €/kWh):** {f_txt(pb2)}")
st.write(f"**Maalämpö C ({h3:.2f} €/kWh):** {f_txt(pb3)}")

# Lainaosuus
lainaosuus = investointi / neliot if neliot > 0 else 0
st.markdown(f"**Lainaosuus investoinnille:** {lainaosuus:,.0f} €/m²")

# PDF
syotteet = {
    "Investointi (€)": investointi,
    "Laina-aika (v)": laina_aika,
    "Korko (%/v)": korko,
    "Sähkönkulutus (kWh/v)": kulutus,
    "Sähkön inflaatio (%/v)": inflaatio,
    "Korjausväli (v)": korjaus_vali,
    "Korjauksen hinta (€)": korjaus_hinta,
    "Korjauslaina (v)": korjaus_laina_aika,
    "Maalämpö kuukausikustannus (€)": maalampo_kk_kulu,
    "Sähköhinta A (€/kWh)": h1,
    "Sähköhinta B (€/kWh)": h2,
    "Sähköhinta C (€/kWh)": h3,
    "Kaukolämpö (€)": kl0,
    "Kaukolämmön inflaatio (%/v)": kl_inf,
    "Neliöt (m²)": neliot,
    "Aikajänne (vuotta)": vuodet,
    "vuosittainen_df": vuosittainen_df,  # PDF: vastiketaulukko
    "saasto_df": saasto_df,              # PDF: säästötaulukko
}

pdf = luo_pdf(fig, lainaosuus, syotteet, vuodet_teksti)
st.download_button("📄 Lataa PDF-raportti", data=pdf, file_name="lämmitysvertailu.pdf", mime="application/pdf")
