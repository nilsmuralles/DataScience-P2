# Parte 1. Contexto y planteamiento

> Reto DFL – Bundesliga Data Shootout: detección de eventos en video de fútbol de la liga alemana.

## Situación problemática

Las ligas de fútbol profesional trabajan con dos tipos de datos. Los de posición dicen dónde está cada jugador y el balón en cada momento, y ya se capturan de forma automática con sistemas de cámaras. Los de evento registran qué acción ocurrió y en qué segundo: un pase, una disputa por el balón, un saque de banda. Esos siguen anotándose a mano, y según la propia Deutsche Fußball Liga (DFL) el proceso requiere varias personas y varias etapas de revisión por cada partido.

Eso lo vuelve lento y caro. Solo las ligas grandes pueden costearlo, las divisiones inferiores se quedan sin esa información, y como depende de personas, la marca de tiempo de cada evento varía según quién la anote. Lo llamativo es que la tecnología ya resolvió la mitad del problema: sabe seguir a los jugadores, pero no interpretar qué están haciendo. Para cerrar esa brecha, la DFL publicó en Kaggle el reto Bundesliga Data Shootout, con video anotado de partidos reales, pidiendo detectar automáticamente tres acciones: el pase (play), la disputa entre dos rivales por el balón (challenge) y el saque de banda (throw-in).

## Problema científico

El problema es determinar si se pueden detectar y clasificar automáticamente los tres eventos de juego (pase, disputa y saque de banda), ubicándolos con precisión en el tiempo, usando únicamente el video de cámara táctica de partidos de la Bundesliga, y con qué nivel de confiabilidad.

Conviene aclarar primero cómo están definidos y anotados esos eventos. El pase es el intento de mandarle el balón a un compañero, aunque falle o lo intercepten, e incluye centros y balones puestos en juego desde tiro libre o córner; es el evento más común con diferencia. La disputa es cuando dos jugadores de equipos contrarios pelean el balón al mismo tiempo. El saque de banda es la reanudación desde la línea lateral, el más escaso de los tres. Cada acción no se marca con una sola línea: viene encerrada en una ventana con un marcador de inicio y otro de fin, así que buena parte de las filas del archivo de anotaciones no son eventos sino delimitadores.

Dicho de otro modo, a partir de la grabación de una mitad de partido hay que producir una lista de eventos con su instante, su clase y qué tan seguro está el modelo, de forma que se parezca a lo que anotaría una persona. Tres cosas hacen que esto sea difícil, y son justamente las que el análisis exploratorio busca dimensionar.

Primero, la precisión temporal que se exige. La métrica original del reto acepta un margen de entre 0.15 y 0.35 segundos para pases y saques de banda, y de entre 0.30 y 0.70 segundos para disputas. A 25 cuadros por segundo, medio segundo son apenas 12 frames, así que no basta con saber que hubo un pase: hay que decir casi exactamente cuándo.

Segundo, el desbalance entre clases. Los pases son muchísimo más frecuentes que las disputas y, sobre todo, que los saques de banda: hay alrededor de 21 pases por cada saque de banda. Con esa proporción, un modelo que respondiera siempre "pase" acertaría más del 80 % de las veces sin haber aprendido nada, lo que vuelve inútil la exactitud global como medida de éxito.

Tercero, lo parecidos que se ven los frames entre sí. El video viene de una cámara táctica fija, con vista alta y lateral: los jugadores miden entre 15 y 20 píxeles de alto y el balón muchas veces ni se distingue. Lo único que cambia entre un pase y un forcejeo es la posición de unos pocos jugadores, así que la señal está en el movimiento local y la cercanía entre ellos, no en propiedades globales de la imagen como el brillo o el color.

De ahí sale la pregunta que guía el trabajo: ¿qué se puede extraer del video y de las anotaciones que realmente permita distinguir entre los tres tipos de evento, y qué limitaciones de los datos disponibles condicionan que un modelo así sea viable?

## Objetivos

### Objetivo general

Hacer un análisis exploratorio de las anotaciones y del video del reto DFL – Bundesliga Data Shootout para caracterizar la estructura, la calidad y la capacidad discriminante de los datos, y así establecer con evidencia qué condiciones tendría que cumplir un modelo de detección automática de eventos.

### Objetivos específicos

1. Describir la estructura y la distribución de las anotaciones sobre los 11,218 registros de los 12 videos: cuántas observaciones y variables hay, de qué tipo es cada una, qué función cumplen los marcadores de inicio y fin, qué valores faltan, y cómo se reparten las clases de evento en el tiempo (eventos por minuto, separación entre eventos y duración de las jugadas).

2. Determinar si las características que se pueden medir sobre los frames (brillo, contraste, saturación, proporción de píxeles de campo, densidad de bordes y un proxy de movimiento entre frames consecutivos) sirven para separar los tres tipos de evento, y documentar las limitaciones de cobertura del conjunto: qué porcentaje de los eventos anotados tiene video disponible, qué sesgo introduce esa pérdida y qué implica el desbalance de clases para el diseño de un futuro modelo.

## Referencias

- DFL – Bundesliga Data Shootout. Kaggle. https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout
- Deutsche Fußball Liga. *Bundesliga Data Shootout: Machine Learning competition on football data.* https://www.dfl.de/en/innovation/bundesliga-data-shootout-machine-learning-competition-on-football-data/
- Deutsche Fußball Liga. *DFL launches "Bundesliga Data Shootout" to unearth new solutions for automated data recording.* https://www.dfl.de/en/news/dfl-launches-bundesliga-data-shootout-to-unearth-new-solutions-for-automated-data-recording/

---

## Notas para la presentación (no va en el informe)

Material de respaldo por si toca explicar por qué el problema es difícil:

- Esto entra en lo que se conoce como *temporal action localization*: no alcanza con clasificar un clip, hay que señalar el segundo exacto dentro de una grabación de casi una hora.
- Los enfoques típicos combinan una lectura espacial de cada frame (redes convolucionales, o detectores tipo YOLO ajustados a fútbol para ubicar jugadores y balón) con algo que modele el tiempo: redes convolucionales 3D, arquitecturas de dos flujos que separan apariencia y movimiento, o transformers sobre ventanas de varios frames.
- El detector genérico de personas de OpenCV (HOG) no funciona aquí: su ventana mínima es de 64×128 px y los jugadores miden ~18 px. Se necesitaría un detector especializado.
