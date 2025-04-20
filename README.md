![qualogo](https://github.com/user-attachments/assets/2b7d1ea1-768c-40fb-8862-5c1b48abc751)
# Quasys

quantum system simulation

* Es importante señalar que esto es una simulación de computación cuántica, no una implementación real en hardware cuántico. El código modela comportamientos cuánticos como la coherencia, interferencia y estados cuánticos usando matemáticas complejas, pero se ejecuta en hardware clásico. 

El código implementa un sistema híbrido de inteligencia artificial que combina:

Una red de nodos fotónicos simulados (para el procesamiento de información cuántica)
Redes neuronales recurrentes (GRU o LSTM) para procesamiento secuencial
Inferencia bayesiana para la toma de decisiones y medición de confianza

Componentes Principales
1. StatisticalAnalysis
Esta clase proporciona métodos estáticos para análisis estadístico:

shannon_entropy: Calcula la entropía de Shannon de una distribución de probabilidades
mahalanobis_distance: Calcula la distancia de Mahalanobis (útil para detectar outliers)
cosine_similarity: Calcula la similitud del coseno entre dos vectores

2. NodoFotonico
Simula un nodo que procesa información cuántica:

Mantiene un estado cuántico como un vector complejo normalizado
Implementa inferencia bayesiana para actualizar sus creencias
Actualiza su estado basado en información de entrada
Predice estados cuánticos futuros basados en entradas, entropía y coherencia

3. RedNodosFotonicos
Implementa una red de nodos fotónicos:

Gestiona múltiples nodos fotónicos interconectados
Propaga información a través de la red simulando una forma de teleportación cuántica
Calcula medidas de coherencia global de la red

4. RNNCoordinator
Coordina modelos de RNN (Redes Neuronales Recurrentes):

Crea y entrena modelos RNN (GRU o LSTM)
Prepara datos secuenciales para entrenamiento
Combina predicciones de RNN con un modelo lineal
Normaliza datos para un rendimiento óptimo

5. ArquitecturaIACuanticaConRNN
Esta es la clase principal que integra todos los componentes:

Combina la red de nodos fotónicos con el coordinador RNN
Procesa datos primero a través de la red cuántica y luego a través de la RNN
Entrena el sistema completo
Realiza predicciones combinando ambos enfoques
Evalúa el rendimiento del modelo
Visualiza resultados y métricas

6. Funciones auxiliares

generar_datos_ejemplo: Crea datos sintéticos para probar el sistema
main: Función principal que ejecuta una demostración del sistema

Flujo de Trabajo

Se generan datos de ejemplo (sinusoidales o aleatorios)
Se dividen en conjuntos de entrenamiento y prueba
Se crea la arquitectura híbrida
Los datos se procesan primero a través de la red de nodos fotónicos
Los resultados del procesamiento cuántico se utilizan para entrenar una RNN
El sistema realiza predicciones combinando ambos enfoques
Se evalúa el rendimiento y se visualizan los resultados

Aspectos Destacados

Enfoque híbrido: Combina conceptos de computación cuántica con aprendizaje profundo
Inferencia bayesiana: Utiliza conceptos de estadística bayesiana para cuantificar incertidumbre
Visualización: Incluye herramientas para visualizar resultados y comportamiento del sistema
Regularización: Implementa técnicas como dropout y batch normalization para prevenir el sobreajuste
Adaptabilidad: Ajusta automáticamente parámetros como la tasa de aprendizaje
