# Resultados. Descripción de la aplicación

Para probar los modelos fuera de los notebooks se construyó una aplicación web en Streamlit. En ella cualquier persona puede subir un frame de un partido y ver qué evento predice cada modelo. También puede explorar los datos con los que se entrenaron y comparar el rendimiento de los algoritmos con gráficas interactivas. Se corre desde la raíz del repositorio con `streamlit run app/app.py` y queda organizada en tres pestañas.

## Explorar datos

La primera pestaña muestra el conjunto de datos tal como quedó después de la limpieza: 9,133 frames muestreados de 3 mitades de partido, cada uno con su clase y las cinco propiedades de imagen que reciben los modelos clásicos. Esas propiedades son brillo, contraste, saturación, proporción de verde y densidad de bordes. Arriba hay tres filtros, que son las clases, los videos y la propiedad a analizar, y todas las gráficas de la pestaña responden a ellos.

Lo primero que se ve es el conteo de frames por clase y la composición de clases dentro de cada video. Ahí aparece el hallazgo que más pesa en todo el proyecto: *play* concentra cerca del 78 % de los frames, mientras que *throwin* apenas llega al 3 %. La aplicación lo señala con un aviso, porque explica por qué después se comparan los modelos con F1 macro y no con accuracy.

Debajo hay un histograma y un diagrama de caja de la propiedad elegida, separados por clase. Al recorrer las cinco propiedades se ve que las cajas de las cuatro clases se traslapan casi por completo. La aplicación calcula a cuántas desviaciones estándar están las medianas entre sí y lo muestra junto a la gráfica. Es la evidencia visual de que una propiedad global del frame dice poco sobre qué evento está pasando, y anticipa el techo que tienen los modelos clásicos. En secciones desplegables se puede ver además la relación entre dos propiedades en un diagrama de dispersión y una tabla de estadísticas descriptivas por clase. Al final hay una galería con frames de ejemplo de cada clase.

> **[Figura A1]** Captura de la pestaña *Explorar datos*: conteo por clase, composición por video e histograma/boxplot de `green_ratio`.

## Clasificar

En la segunda pestaña el usuario sube una imagen (JPG, PNG, BMP o WEBP) y decide si clasificarla con todos los modelos o solo con los que elija. El preprocesamiento pasa por completo en segundo plano. La imagen se decodifica igual que en el entrenamiento y se reescala a 320 píxeles de ancho. Luego se calculan las cinco propiedades con la misma función que generó `frame_metrics.csv` (`src/features.py`). Para el modelo convolucional, en cambio, se extrae el vector de características de la ResNet18 (`src/cnn_features.py`). El usuario no ve nada de esto: solo sube la imagen y recibe el resultado.

Por cada modelo se muestra una tarjeta con la clase predicha, la confianza y el F1 macro que ese modelo obtuvo en los datos de prueba. Así el usuario puede juzgar cuánto confiar en cada respuesta. Cuando se usan varios modelos, la aplicación indica también cuál fue la clase más votada. Una gráfica de barras apiladas compara la probabilidad que cada modelo asigna a las cuatro clases, y debajo está la misma información en una tabla. Con esto se nota cuándo los modelos coinciden y cuándo alguno duda entre dos clases.

> **[Figura A2]** Captura de la pestaña *Clasificar* con un frame subido y la opción *Todos* los modelos.

## Rendimiento

La tercera pestaña reúne los resultados de la evaluación. Lee automáticamente todos los archivos `results/metrics_*.csv` y `results/predictions_*.csv`, así que cualquier modelo nuevo aparece sin tocar el código. Antes de las gráficas se explica cómo se evaluó: dejando un video fuera cada vez, de modo que el modelo nunca ve el partido con el que se le prueba. Un interruptor permite esconder o mostrar todas las gráficas, y un selector deja comparar solo los modelos que interesen.

Las gráficas siguen un orden que cuenta la historia de los resultados:

1. **Accuracy vs. F1 macro**, con la media y la desviación entre videos. El Dummy, que siempre predice *play*, tiene el accuracy más alto (0.78) pero el F1 macro más bajo. La aplicación lo destaca con un aviso de que el accuracy engaña en este problema.
2. **F1 por clase**: todos los modelos son buenos en *play* y casi nulos en *challenge* y *throwin*, que son justo las clases que importan.
3. **F1 macro en cada video de prueba**, para ver qué tan estable es cada modelo de un partido a otro.
4. **Tiempo de entrenamiento y tiempo de predicción frente al F1 macro**, en dos gráficas separadas con escala logarítmica. Sirven para decidir si la mejora de un modelo justifica su costo. Por ejemplo, HistGradientBoosting tarda varias veces más que Random Forest en entrenar y no lo supera.
5. **Matriz de confusión** del modelo y el video elegidos, normalizada por fila para que la diagonal sea el recall de cada clase. Junto a ella, una gráfica compara qué clases predice cada modelo contra la distribución real de etiquetas, y se ve claramente cuánto se cargan las predicciones hacia *play*.

Todas las gráficas muestran el valor exacto al pasar el cursor. Desde la leyenda se puede ocultar un modelo o una clase, y se pueden ampliar o descargar como imagen. Al final hay una tabla con todas las métricas por video.

> **[Figura A3]** Captura de la pestaña *Rendimiento*: accuracy vs. F1 macro y F1 por clase.
>
> **[Figura A4]** Captura de la matriz de confusión y la gráfica de clases predichas.

## Diseño y uso del color

Cada clase tiene un color fijo que se repite en todas las pestañas: ámbar para *bg*, azul para *play*, rojo para *challenge* y verde azulado para *throwin*. Así el usuario aprende la asociación una vez y la reconoce en cualquier gráfica. Los colores se eligieron con matices bien separados en el círculo cromático y se validaron con simulaciones de los tres tipos de daltonismo (deuteranopia, protanopia y tritanopia), tanto en fondo claro como oscuro. Como el color nunca es la única pista, el nombre de la clase siempre aparece escrito en los ejes o en la leyenda.

Los modelos usan una paleta aparte para no confundirse con las clases. El Dummy va en gris a propósito: es la línea base contra la que se comparan los demás, no un competidor. La matriz de confusión usa una escala secuencial de un solo tono (de azul claro a oscuro), que es la forma adecuada de representar magnitudes. El resto de la interfaz se mantiene neutra, con fondo claro, ejes y cuadrículas discretos, para que la atención vaya a los datos.

## Selección de tecnologías

**Streamlit** se eligió porque permite construir la aplicación completa en Python, el mismo lenguaje de todo el proyecto. Los modelos se guardaron como pipelines de scikit-learn con `joblib`, así que la aplicación los carga y usa directamente, sin servidores ni APIs intermedias. También reutiliza la misma función de extracción de características del análisis exploratorio, lo que garantiza que una imagen nueva se preprocese exactamente igual que los datos de entrenamiento. Además trae componentes listos para lo que pedía el proyecto: pestañas, carga de archivos, filtros e interruptores. Su sistema de caché evita volver a leer los datos y los modelos con cada interacción. Frente a alternativas como Flask o Dash, Streamlit necesita mucho menos código para una aplicación de análisis como esta y permite que cualquier integrante del equipo la modifique.

**Plotly** se eligió para las gráficas porque son interactivas de forma nativa: muestran el valor exacto al pasar el cursor, permiten hacer zoom y ocultar series desde la leyenda, y se pueden descargar como imagen. Eso cumple el requisito de mostrar el rendimiento con gráficas interactivas y deja que el usuario explore los resultados por su cuenta. Se integra directamente con Streamlit y con pandas, y permite fijar los colores de cada clase para mantener la paleta consistente. Las gráficas estáticas del informe, en cambio, se hicieron con matplotlib en los notebooks.
