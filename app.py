# app.py — Lämmitysvaihtoehtojen vertailu (A/B/C kuvaajassa, taulukot B, PDF sisältää takaisinmaksuajat)
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

def takaisinmaksuaika_investointi(investointi, kaukolampo_sarja, maalampo_sarja):
    vuosittainen_saasto = np.array(kaukolampo_sarja) - np.array(maalampo_sarja)
    kum = np.cumsum(vuosittainen_saasto)
    for vuosi, summa in enumerate(kum, 1):
        if summa >= investointi:
            return vuosi
    return None

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

def luo_pdf(kaavio, vuosittainen_df, saasto_df, vuodet_teksti, lainaosuus, syotteet, pbA, pbB, pbC, h1, h2, h3):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Lämmitysvaihtoehtojen vertailuraportti", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))
    # Syötetyt arvot
    elements.append(Paragraph("Syötetyt arvot:", styles['Heading2']))
    for nimi, arvo in syotteet.items():
        if isinstance(arvo, (pd.DataFrame, plt.Figure)):
            continue
        elements.append(Paragraph(f"{nimi}: {arvo}", styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))
    # Kuvaaja
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    kaavio.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
    elements.append(Paragraph(f"Lämmityskustannukset {vuodet_teksti} ajalta", styles['Heading2']))
    elements.append(Image(tmpfile.name, width=6*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))
    # Vastiketaulukko (vain B)
    table_data = _format_df_for_pdf(vuosittainen_df)
    table_v = Table(table_data, repeatRows=1)
    table_v.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                 ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                                 ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                 ('FONTSIZE', (0, 0), (-1, -1), 7),
                                 ('ALIGN', (1, 1), (-1, -1), 'RIGHT')]))
    elements.append(Paragraph("Vastikkeet vuoden välein (€/m²/kk) — Vaihtoehto B", styles['Heading2']))
    elements.append(table_v)
    elements.append(Spacer(1, 0.3 * inch))
    # Säästötaulukko
    table_data_s = _format_df_for_pdf(saasto_df)
    table_s = Table(table_data_s, repeatRows=1)
    table_s.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                 ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                                 ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                 ('FONTSIZE', (0, 0), (-1, -1), 7),
                                 ('ALIGN', (1, 1), (-1, -1), 'RIGHT')]))
    elements.append(Paragraph("Vuosittainen säästö (Maalämpö B vs. Kaukolämpö) €/vuosi per asunto", styles['Heading2']))
    elements.append(table_s)
    elements.append(Spacer(1, 0.3 * inch))
    # Takaisinmaksuajat
    f_txt = lambda v: f"{v} vuotta" if v else f"ei {vuodet_teksti} ajalla"
    elements.append(Paragraph("Investoinnin takaisinmaksuaika:", styles['Heading2']))
    elements.append(Paragraph(f"Maalämpö A ({h1:.2f} €/kWh): {f_txt(pbA)}", styles['Normal']))
    elements.append(Paragraph(f"Maalämpö B ({h2:.2f} €/kWh): {f_txt(pbB)}", styles['Normal']))
    elements.append(Paragraph(f"Maalämpö C ({h3:.2f} €/kWh): {f_txt(pbC)}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Lainaosuus: {lainaosuus:,.0f} €/m²", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- SOVELLUS ----------

st.set_page_config(page_title="Lämmitysvaihtoehdot", layout="wide")
st.title("Maalämpö (A/B/C) vs Kaukolämpö — PDF")