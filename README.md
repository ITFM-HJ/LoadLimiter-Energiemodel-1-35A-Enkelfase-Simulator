# LoadLimiter Energiemodel — 1×35A Enkelfase Simulator

**[▶ Open de simulator](https://loadlimiter.streamlit.app) **

---

## Wat is dit?

Dit model simuleert het elektriciteitsverbruik van een **all-electric woning op een enkelfasige 35A-aansluiting** gedurende één dag. Het toont wanneer de fasegrens wordt overschreden en wat het effect is van een **LoadLimiter** — een apparaat dat de warmtepomp tijdelijk onderbreekt bij piekbelasting.

### Het probleem

Nederland elektrificeert massaal. Woningen krijgen warmtepompen, inductiekookplaten en laadpalen — maar de enkelfasige aansluiting (35A = ~8 kW) blijft soms noodgedwongen ongewijzigd. De combinatie van gelijktijdig gebruik kan de zekering laten doorbranden/afschakelen of netcongestie veroorzaken.

Een **LoadLimiter** pauzeert de warmtepomp kort bij piekbelasting (bijv. koken + douchen + EV laden). De warmtevraag wordt "uitgesteld", niet bespaard — het huis koelt iets af en de WP haalt de achterstand later in.

### Wat kun je berekenen?

- Hoe vaak en hoe lang de 35A-grens wordt overschreden per gezinstype
- Wat het effect is van een LoadLimiter op piekbelasting en binnentemperatuur
- Hoeveel een EV kan laden bij een drukke avond (en wat het tekort is bij vertrek)
- Wat er gebeurt bij extreme kou (winter, −8°C) met een slecht geïsoleerde woning
- Monte Carlo: hoe variabel is het over 200 willekeurige dagscenario's?

---

## Snel starten (lokaal)

```bash
git clone https://github.com/ITFM-HJ/LoadLimiter-Energiemodel-1-35A-Enkelfase-Simulator
cd LoadLimiter-Energiemodel-1-35A-Enkelfase-Simulator/energiemodel
pip install -r requirements.txt
streamlit run app_advanced.py
```

Open dan http://localhost:8501 in je browser.

---

## Modelopbouw

```
app_advanced.py          ← dit model (15-min resolutie, thermisch, Monte Carlo)
app.py                   ← origineel deterministisch uurmodel (ongewijzigd)
requirements.txt
.streamlit/config.toml   ← thema-instellingen
```

### Thermisch huismodel

Het model gebruikt een RC-thermisch model (1e orde):

```
q_verlies[t]  = UA × (T_in[t] − T_buiten)          [kW]
q_vraag[t]    = UA × (T_set[t] − T_buiten)          [kW steady-state]
q_herstel[t]  = (T_set − T_in[t]) × C_th / τ        [kW temperatuurherstel]
wp_want[t]    = 0  als T_in ≥ T_set  (WP niet nodig)
              = min(wp_max_elec, (q_vraag + q_herstel) / COP)  anders
T_in[t+1]     = T_in[t] + (wp_want × COP − q_verlies) × Δt / C_th
```

**Parameters (instelbaar in sidebar):**
| Parameter | Beschrijving | Standaard |
|-----------|-------------|----------|
| C_th | Thermische massa woning | 5 kWh/°C |
| UA | Warmtedoorgang | 300 W/°C |
| wp_elec_max | Max. WP elektrisch vermogen | 3,5 kW |
| τ_herstel | Opwarmtijdconstante | 4 uur |

### Bekende vereenvoudigingen

- **Één dag** — geen carry-over van binnentemperatuur of EV-accu tussen dagen
- **Aanloopstromen** niet gemodelleerd — werkelijke pieken kunnen kortstondig hoger liggen
- **Lineair COP-model** — echte WP-prestaties wijken per fabrikant/model af
- **EV-headroom** gebruikt een profielbased WP-referentie, niet de thermisch berekende waarde
- **UA en C_th** zijn woningspecifiek — kalibratie op meetdata geeft nauwkeurigere resultaten

---

## Feedback geven

Gevonden iets wat niet klopt? Suggestie voor verbetering? Open een [GitHub Issue](../../issues).

Specifieke vragen waarop we feedback zoeken:
- Kloppen de huishoudprofielen (Senioren / Jonge kinderen / Tieners) met de praktijk?
- Zijn de thermische parameters (UA, C_th) realistisch voor een typische NL tussenwoning?
- Welke scenario's missen er nog?

---

## Context

Dit model is ontwikkeld als onderdeel van een proof-of-concept voor een **LoadLimiter-product** voor
sociale woningbouw in Nederland. Het probleem: woningen gaan all-electric terwijl de enkelfase-
aansluiting ongewijzigd blijft. Netcongestie en kapotte zekeringen zijn het gevolg.

Het model helpt om de businesscase en technische keuzes te onderbouwen — geen garanties op exactheid,
wel een eerlijk inzicht in de orde van grootte van het probleem.

---

## Licentie

MIT — vrij te gebruiken, aanpassen en verspreiden. Naamsvermelding op prijs gesteld.

---

*Model gebouwd met [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com) · [NumPy](https://numpy.org)*
