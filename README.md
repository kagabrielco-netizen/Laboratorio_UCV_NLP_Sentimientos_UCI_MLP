# Laboratorio NLP - Análisis de Sentimientos (TF-IDF + MLP)

## Descripción
Sistema de análisis de sentimientos usando NLP clásico (TF-IDF) y un 
Perceptrón Multicapa (MLP) para clasificar comentarios como positivos o 
negativos.

## Dataset
UCI ML Repository - Sentiment Labelled Sentences (Amazon, IMDb, Yelp).
Total: 2748 comentarios etiquetados.

## Exploración inicial
<img width="228" height="66" alt="image" src="https://github.com/user-attachments/assets/b7cdbcf4-e1ca-4c04-a152-a1d5bd83b1a6" />

## Distribución de sentimientos
<img width="432" height="294" alt="image" src="https://github.com/user-attachments/assets/8c00f387-722e-4506-8ca3-bab2dea2556a" />

## Limpieza de texto
<img width="562" height="267" alt="image" src="https://github.com/user-attachments/assets/dae6224b-f34e-4481-982a-99ea50841019" />

## Vectorización TF-IDF

<img width="218" height="30" alt="image" src="https://github.com/user-attachments/assets/01433a43-a232-489b-abfe-fab44138f1d4" />



<img width="161" height="477" alt="image" src="https://github.com/user-attachments/assets/f5511b9e-f713-4d55-b6a5-50d5773eca0c" />

## Entrenamiento del modelo base (64, 32)
<img width="123" height="57" alt="image" src="https://github.com/user-attachments/assets/a089e36a-9a7d-493e-a1d7-0b7ed72b9502" />

<img width="325" height="116" alt="image" src="https://github.com/user-attachments/assets/f2ad371f-d0c3-4d75-9dea-f26b44030b68" />

<img width="447" height="342" alt="image" src="https://github.com/user-attachments/assets/5950ab2c-f83e-4a7a-ab64-509b048829fd" />

<img width="543" height="353" alt="image" src="https://github.com/user-attachments/assets/e09ce724-8bf3-4ea9-a9e7-aa10536f0d2a" />


## Predicciones con comentarios nuevos (modelo base)
<img width="717" height="154" alt="image" src="https://github.com/user-attachments/assets/05e30e2b-6d81-41dd-8b7f-d2c3ce0f38af" />


## Análisis de errores (modelo base)
<img width="366" height="262" alt="image" src="https://github.com/user-attachments/assets/0c8618e5-9be4-4866-8f80-99234d54c5d3" />


## Reto: Comparación de arquitecturas MLP

### Tabla comparativa

| Arquitectura | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| (16,) | 0.7939 | 0.8106 | 0.7716 | 0.7906 |
| (64, 32) | 0.7891 | 0.7630 | 0.8438 | 0.8014 |
| (128, 64, 32) | 0.7842 | 0.8199 | 0.7332 | 0.7741 |

### Arquitectura elegida: (64, 32)

Aunque la arquitectura (128, 64, 32) es la más grande, no obtuvo el mejor 
desempeño: su recall (0.733) fue el más bajo de las tres, mientras que 
(64, 32) logró el mejor F1-score (0.801) y mejor recall (0.844), detectando 
más comentarios positivos correctamente. Esto confirma que un modelo más 
grande no siempre mejora el rendimiento: puede sobreajustarse a patrones 
específicos del entrenamiento sin generalizar mejor.

### Prueba con 10 comentarios nuevos (arquitectura 128,64,32)

Resultado: 7/10 aciertos (70%).

**Errores identificados:**
1. "This restaurant exceeded all my expectations" → predicho Negativo, 
   esperado Positivo.
2. "Amazing customer support, very helpful" → predicho Negativo, esperado 
   Positivo.
3. "Nothing special, quite forgettable" → predicho Positivo, esperado 
   Negativo.

### ¿El modelo entiende el lenguaje o solo reconoce patrones estadísticos?

El modelo solo reconoce patrones estadísticos de palabras, no comprende el 
lenguaje. Esto se evidencia en frases como "Amazing customer support, very 
helpful", claramente positiva para cualquier persona, pero que el modelo 
clasificó como negativa — probablemente porque palabras como "support" 
aparecen en contextos negativos dentro del corpus de entrenamiento (ej. 
quejas de servicio al cliente). De igual forma, "Nothing special, quite 
forgettable" es una crítica sutil sin palabras negativas explícitas, y el 
modelo no logra captar ese matiz porque TF-IDF no entiende sarcasmo, tono 
ni contexto, solo cuenta la frecuencia de palabras.

### Pregunta central

> Si usted fuera responsable de este sistema, ¿confiaría en él para tomar 
decisiones automáticas de negocio? Justifique.

No confiaría en él para decisiones automáticas sin supervisión humana. Con 
un 70-80% de accuracy y errores en casos donde el sentimiento es claro para 
una persona, el modelo podría malinterpretar quejas reales como positivas 
(o viceversa), afectando decisiones de negocio. Sería más apropiado usarlo 
como una herramienta de apoyo para priorizar revisión manual, no como 
decisor final.

## Preguntas de análisis

1. **¿Por qué una computadora no entiende texto directamente?**
Porque los algoritmos trabajan con números, no con palabras; el texto debe convertirse primero en vectores numéricos.

2. **¿Qué es TF-IDF?**
Una técnica que asigna un peso a cada palabra según su frecuencia en un documento y su rareza en el resto del corpus, destacando términos relevantes.

3. **¿Qué diferencia existe entre tokenizar y vectorizar?**
Tokenizar es dividir el texto en unidades (palabras); vectorizar es convertir esas unidades en números que el modelo pueda procesar.

4. **¿Qué palabras parecen más importantes para clasificar sentimientos?**
Palabras con carga emocional clara como "great", "excellent", "terrible" o "awful", según lo mostrado en la tabla de términos TF-IDF más relevantes.

5. **¿Qué métrica considera más importante en este caso: accuracy, precision, recall o F1-score?**
El F1-score, porque equilibra precision y recall, siendo útil cuando ambos tipos de error (falsos positivos y negativos) importan para el negocio.

6. **¿Qué errores cometió el modelo?**
Clasificó como negativos comentarios claramente positivos (ej. sobre buen servicio) y como positivo un comentario negativo sutil sin palabras negativas explícitas.

7. **¿Por qué detectar sarcasmo es difícil para este enfoque?**
Porque TF-IDF solo cuenta palabras, sin considerar el tono, contexto o intención real detrás de ellas.

8. **¿Qué ocurriría si ingresáramos un comentario en otro idioma?**
El modelo lo clasificaría mal o de forma aleatoria, ya que el vectorizador y el entrenamiento solo reconocen vocabulario en inglés.

## Conclusión

El sistema de análisis de sentimientos con TF-IDF y MLP logró un desempeño 
aceptable (78-79% de accuracy), pero mostró limitaciones claras al 
enfrentar comentarios con matices, ironía o vocabulario fuera de su 
contexto de entrenamiento.

Al comparar tres arquitecturas, la intermedia (64, 32) tuvo el mejor 
balance entre precision y recall, superando incluso a la arquitectura más 
grande (128, 64, 32), lo que confirma que aumentar el tamaño del modelo no 
garantiza mejores resultados.

Las pruebas con comentarios nuevos (7/10 aciertos) evidencian que el modelo 
reconoce patrones estadísticos de palabras, no el significado real del 
lenguaje, justificando por qué enfoques más avanzados como Transformers son 
necesarios para tareas de NLP más complejas.
