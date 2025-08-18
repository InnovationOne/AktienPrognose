# KI‑Aktienprognose – Setup, Start & Updates (Windows)

Diese README beschreibt die **Ersteinrichtung** mit Conda, den **Start** des Projekts und **Updates**. 
Beispielpfad: `C:\Users\<UserName>\Desktop\AktienPrognose`.

---

## TL;DR
1. **Anaconda installieren**. Nutze die **Anaconda Prompt**.
2. **Umgebung erstellen & aktivieren:**
   `conda env create -f environment.yml`
   `conda activate ki-aktienprognose-env`
   `python -m pip install --upgrade pip`
3. **Starten:**
   `jupyter lab`

---

## Ersteinrichtung (Schritt für Schritt)

1. **Anaconda Prompt** öffnen.
2. In den **Projektordner** wechseln, z. B.:
   `cd C:\Users\<UserName>\Desktop\AktienPrognose`
3. **Umgebung erzeugen**:
   `conda env create -f environment.yml`
4. **Aktivieren**:
   `conda activate ki-aktienprognose-env`
5. **Pip aktualisieren**:
   `python -m pip install --upgrade pip`
6. **Sanity‑Checks**:
   `python -c "import tensorflow as tf; print(tf.__version__)"`
   `python -c "import yfinance as y; print('yfinance ok')"`

---

## Start nach der Einrichtung

- **JupyterLab starten**:
  `jupyter lab`
  > Browser öffnet sich automatisch.

---

## Updates & Pflege

### A) Pakete innerhalb der bestehenden Umgebung aktualisieren
> Vorher die Umgebung aktivieren:
  `conda activate ki-aktienprognose-env`

- **Conda selbst aktualisieren**:
  `conda update -n base -c defaults conda`
- **Conda‑Pakete im Env aktualisieren**:
  `conda update --all`
- **Pip‑Pakete aktualisieren** (TensorFlow, yfinance):
  `pip install -U tensorflow yfinance`
- **Integritätscheck** (zeigt kaputte Abhängigkeiten):
  `pip check`

### B) Änderungen an `environment.yml` übernehmen
Wenn man die `environment.yml` anpasst.
`conda activate ki-aktienprognose-env`
`conda env update -f environment.yml --prune`

### C) Environment neu bauen (sauberer Reset)
`conda env remove -n ki-aktienprognose-env`
`conda env create -f environment.yml`

---

## Nützliche Befehle (Cheat‑Sheet)

`conda info --envs`                         # Alle Umgebungen anzeigen
`conda list`                                # Pakete in aktiver Umgebung
`pip list`                                  # Pip‑Pakete in aktiver Umgebung
`pip freeze > requirements.txt`             # (optional) exakte Pip‑Versionen exportieren
`conda clean -a`                            # Cache aufräumen (bei Solver-Problemen hilfreich)
`conda env remove -n ki-aktienprognose-env` # Umgebung löschen