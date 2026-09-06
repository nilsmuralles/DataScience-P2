# DataScience-P2 — Detección de pases (DFL Bundesliga)

Análisis exploratorio para el Reto 21: detección de pases, throw-ins y challenges en
videos de la Bundesliga (competencia Kaggle *DFL - Bundesliga Data Shootout*).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_extraction.ipynb   # Run All
```

## Datos

La competencia está cerrada y ya no deja descargar los datos originales. Se usan mirrors:

| Qué | Fuente | Peso | ¿En el repo? |
|---|---|---|---|
| **Anotaciones** `data/raw/train_annotations.csv` | HuggingFace `dbal0503/Bundesliga` (`train.csv`) | 460 KB | **Sí** |
| **Videos** | Kaggle `saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv` | ~10 GB | No |

- Las anotaciones cubren **12 videos / 11.218 eventos**. Es lo único que necesitan las
  Personas 1, 2 y 3 (ya está versionado).
- El mirror de videos trae **5 de los 12** videos, cada mitad partida en clips de ~30 s.
  Solo lo necesita la Persona 4. No pasarlo por WhatsApp: que cada quien lo baje con

  ```bash
  kaggle auth login   # o export KAGGLE_API_TOKEN=xxxx
  kaggle datasets download -d saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv -p data/raw --unzip
  ```

  Queda en `data/raw/DFL Bundesliga Data Shootout/train/<PREFIJO><video_id>/... (N).mp4`.

## Qué produce el notebook 01 (en `data/processed/`, todo se versiona menos `frames/`)

| Archivo | Contenido |
|---|---|
| `events.csv` | anotaciones normalizadas: 1 fila por evento, atributos explotados, clip donde cae |
| `plays.csv` | intervalos de jugada (`start`/`end`) con duración |
| `video_meta.csv` | 1 fila por video: nº de clips, fps, resolución, conteo de eventos |
| `clip_index.csv` | mapa clip → (video, rango de segundos, ruta) |
| `frames_index.csv` | frames muestreados para el análisis visual (los `.jpg` NO se versionan) |
