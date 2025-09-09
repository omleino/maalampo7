# app.py — Lämmitysvaihtoehtojen vertailu (horisontti = laina-aika + 1) + PDF-raportti
# - Vuosittainen vastiketaulukko: vain vaihtoehto B (+ kaukolämpö)
# - Säästö (B vs KL) per asunto skaalattu suhteessa talon kokonaisneliöihin
# - PDF: samat taulukot ja 2 desimaalin numeromuoto
# - Takaisinmaksuaika ei tulostu PDF:ään (vain UI:ssa)

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

# ... (rest of the code continues)
