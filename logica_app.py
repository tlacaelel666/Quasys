
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import logging
from typing import Tuple, Dict, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from scipy.stats import beta as beta_dist

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalAnalysis:
    """Clase para análisis estadístico de datos y estados"""
    
    @staticmethod
    def shannon_entropy(probabilities):
        """Calcula la entropía de Shannon de una distribución de probabilidades"""
        # Filtrar valores que sean cero para evitar log(0)
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0
    
    @staticmethod
    def mahalanobis_distance(x, mean, cov):
        """Calcula la distancia de Mahalanobis entre un punto y una distribución"""
        x_minus_mean = x - mean
        inv_cov = np.linalg.inv(cov)
        return np.sqrt(np.dot(np.dot(x_minus_mean, inv_cov), x_minus_mean.T))
    
    @staticmethod
    def cosine_similarity(a, b):
        """Calcula la similitud del coseno entre dos vectores"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class NodoFotonico:
    """
    Simula un nodo fotónico que procesa información cuántica
    y aplica inferencia bayesiana para la toma de decisiones.
    """
    def __init__(self, id, dim=2, learning_rate=0.1):
        self.id = id
        self.dimension = dim
        # Estado cuántico inicial (vector complejo normalizado)
        self.estado = np.random.rand(dim) + 1j * np.random.rand(dim)
        self.estado = self.estado / np.linalg.norm(self.estado)
        # Parámetros para inferencia bayesiana
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        self.learning_rate = learning_rate
        self.adaptative_lr_decay = 0.99
        self.initial_learning_rate = learning_rate
        self.coherence_history = []
    
    def actualizar_estado(self, input_estado):
        """Actualiza el estado del nodo basado en entrada y coherencia"""
        # Simula interferencia cuántica
        nuevo_estado = self.estado + 0.5 * input_estado
        self.estado = nuevo_estado / np.linalg.norm(nuevo_estado)
        
        # Calcular coherencia del estado actualizado (simplificación)
        coherence = np.abs(np.mean(self.estado))
        self.coherence_history.append(coherence)
        
        return self.estado
    
    def inferencia_bayesiana(self, observacion):
        """Actualiza conocimiento usando inferencia bayesiana"""
        # Actualiza parámetros de distribución beta (modelo bayesiano simple)
        if observacion == 1:
            self.prior_alpha += 1
        else:
            self.prior_beta += 1
        
        # Calcula probabilidad posterior
        prob = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return prob
    
    def predict_quantum_state(self, input_state, entropy, coherence):
        """
        Predice el próximo estado cuántico basado en entrada, entropía y coherencia.
        
        Args:
            input_state: Estado de entrada (np.array o tensor)
            entropy: Entropía del sistema
            coherence: Medida de coherencia cuántica
        
        Returns:
            Tuple: (nuevo estado, posterior)
        """
        # Convertir a array si es tensor
        if isinstance(input_state, tf.Tensor):
            input_array = input_state.numpy()
        else:
            input_array = np.array(input_state)
        
        # Reshape si es necesario (asumiendo que vendrá como [batch, features])
        if input_array.ndim > 1:
            input_vector = input_array.reshape(-1)
        else:
            input_vector = input_array
            
        # Normalizar input vector
        input_vector = input_vector / (np.linalg.norm(input_vector) + 1e-10)
        
        # Crear un vector de fase basado en la entropía y coherencia
        phase_factor = np.exp(1j * np.pi * (entropy / (coherence + 1e-10)) * np.random.rand(len(input_vector)))
        
        # Aplicar fase al input
        phased_state = input_vector * phase_factor
        
        # Calcular el "posterior" como una medida de calidad de la predicción
        posterior = np.exp(-entropy) * coherence
        
        # Devolver como tensor para compatibilidad
        return tf.convert_to_tensor(phased_state, dtype=tf.complex64), posterior
    
    def update_learning_rate(self):
        """Actualiza la tasa de aprendizaje de forma adaptativa"""
        self.learning_rate *= self.adaptative_lr_decay
        self.learning_rate = max(self.learning_rate, 0.01)  # Asegurarse que no sea demasiado pequeña


class RedNodosFotonicos:
    """Implementa una red de nodos fotónicos para procesamiento distribuido"""
    def __init__(self, num_nodos=3, dimension=2):
        self.nodos = [NodoFotonico(i, dimension) for i in range(num_nodos)]
        self.matriz_conexiones = np.random.rand(num_nodos, num_nodos)
        np.fill_diagonal(self.matriz_conexiones, 0)  # No auto-conexiones
    
    def propagar_informacion(self, input_datos):
        """Propaga información a través de la red de nodos"""
        estados_resultantes = []
        
        # Inicializar estados de entrada
        for i, nodo in enumerate(self.nodos):
            if i < len(input_datos):
                # Codificar entrada en estado cuántico
                estado_entrada = np.array(input_datos[i])
                if estado_entrada.ndim == 0:  # Si es un escalar
                    estado_entrada = np.array([np.sqrt(1-estado_entrada**2), estado_entrada])
                estados_resultantes.append(nodo.actualizar_estado(estado_entrada))
            else:
                estados_resultantes.append(nodo.estado)
        
        # Simular comunicación entre nodos (teleportación cuántica simplificada)
        for _ in range(3):  # Iteraciones de propagación
            nuevos_estados = []
            for i, nodo in enumerate(self.nodos):
                # Calcula influencia ponderada de otros nodos
                estado_combinado = np.zeros(nodo.dimension, dtype=complex)
                for j, otro_nodo in enumerate(self.nodos):
                    if i != j:
                        peso = self.matriz_conexiones[i, j]
                        estado_combinado += peso * estados_resultantes[j]
                
                # Normalizar y actualizar
                if np.linalg.norm(estado_combinado) > 0:
                    estado_combinado = estado_combinado / np.linalg.norm(estado_combinado)
                    nuevos_estados.append(nodo.actualizar_estado(estado_combinado))
                else:
                    nuevos_estados.append(nodo.estado)
            
            estados_resultantes = nuevos_estados
            
        return estados_resultantes
    
    def calcular_coherencia_global(self):
        """Calcula una medida de coherencia global de la red"""
        coherencia_total = 0.0
        for nodo in self.nodos:
            # Tomamos el último valor de coherencia registrado
            if nodo.coherence_history:
                coherencia_total += nodo.coherence_history[-1]
        return coherencia_total / len(self.nodos) if self.nodos else 0.0


class RNNCoordinator:
    """
    Coordinador para modelos RNN que se integra con la red de nodos fotónicos.
    
    Gestiona el entrenamiento, predicción y el guardado de modelos híbridos.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 rnn_type: str = 'GRU', dropout_rate: float = 0.2, use_batch_norm: bool = True):
        """
        Inicializa el coordinador con parámetros de modelo.

        Args:
            input_size (int): Tamaño de entrada.
            hidden_size (int): Tamaño de la capa oculta.
            output_size (int): Tamaño de salida.
            rnn_type (str): Tipo de RNN a usar ('GRU' o 'LSTM').
            dropout_rate (float): Tasa de dropout para regularización.
            use_batch_norm (bool): Si se debe usar BatchNormalization.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.scaler = StandardScaler()  # Normalización de datos robusta a outliers
        self.linear_model = Ridge(alpha=1.0)  # Regularización para evitar overfitting
        self.rnn_model = None

    def create_rnn_model(self) -> tf.keras.Model:
        """
        Crea y compila un modelo RNN con capas GRU/LSTM, Dropout y BatchNormalization.

        Returns:
            tf.keras.Model: Modelo RNN compilado.
        """
        model = Sequential()
        if self.rnn_type == 'GRU':
            model.add(GRU(self.hidden_size,
                          input_shape=(None, self.input_size),
                          return_sequences=True))
        elif self.rnn_type == 'LSTM':
            model.add(LSTM(self.hidden_size,
                          input_shape=(None, self.input_size),
                          return_sequences=True))
        else:
            raise ValueError("rnn_type debe ser 'GRU' o 'LSTM'")

        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        if self.rnn_type == 'GRU':
            model.add(GRU(self.hidden_size // 2))
        else:  # LSTM
            model.add(LSTM(self.hidden_size // 2))

        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.output_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        self.rnn_model = model
        return model

    def prepare_data(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos secuenciales para el entrenamiento de la RNN.

        Args:
            data (np.ndarray): Datos de entrada.
            sequence_length (int): Longitud de la secuencia.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Datos X e y preparados.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     sequence_length: int, epochs: int = 100, batch_size:int = 32) -> None:
        """
        Entrena los modelos RNN y la regresión lineal.
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento.
            y_train (np.ndarray): Etiquetas de entrenamiento.
            sequence_length (int): Longitud de la secuencia.
            epochs (int, optional): Número de épocas de entrenamiento. Defaults to 100.
            batch_size (int, optional): Tamaño del lote para el entrenamiento
        """
        # Preparar datos para RNN
        X_rnn, y_rnn = self.prepare_data(X_train, sequence_length)
        
        if len(X_rnn) == 0 or len(y_rnn) == 0:
            logger.warning("No hay suficientes datos para entrenar después de preparar las secuencias.")
            return
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_scaled = self.scaler.transform(y_train)  # Escalar tambien y_train
        
        # Entrenar la RNN
        if self.rnn_model is None:
            self.rnn_model = self.create_rnn_model()
            
        if len(X_rnn) > 0 and len(y_rnn) > 0:
            self.rnn_model.fit(X_rnn, y_rnn, epochs=epochs, verbose=1, batch_size=batch_size)
        
        # Entrenar la regresión lineal
        self.linear_model.fit(X_train_scaled, y_train_scaled)

    def predict(self, X: np.ndarray, sequence_length: int, rnn_weight: float = 0.7) -> np.ndarray:
        """
        Realiza predicciones combinando los modelos RNN y de regresión lineal.

        Args:
            X (np.ndarray): Datos de entrada.
            sequence_length (int): Longitud de la secuencia.
            rnn_weight (float, optional): Peso del modelo RNN en la combinación. Defaults to 0.7.

        Returns:
            np.ndarray: Predicciones combinadas.
        """
        if len(X) <= sequence_length:
            logger.warning("Los datos proporcionados son más cortos que sequence_length. Ajustando.")
            sequence_length = max(1, len(X) - 1)
            
        X_rnn, _ = self.prepare_data(X, sequence_length)
        
        if X_rnn.size == 0:
            logger.warning("No hay suficientes datos para hacer predicciones. Devolviendo array vacío.")
            return np.array([])
            
        X_scaled = self.scaler.transform(X)

        rnn_pred = self.rnn_model.predict(X_rnn)
        # Para la regresión lineal, se descartan las primeras "sequence_length" predicciones
        linear_pred = self.linear_model.predict(X_scaled)[sequence_length:]
        
        # Verificar que ambas predicciones tienen la misma forma
        if len(rnn_pred) != len(linear_pred):
            min_len = min(len(rnn_pred), len(linear_pred))
            rnn_pred = rnn_pred[:min_len]
            linear_pred = linear_pred[:min_len]
        
        # Combinar predicciones y desescalar
        combined_pred_scaled = rnn_weight * rnn_pred + (1 - rnn_weight) * linear_pred
        
        # Dado que y_train fue escalado, ahora combined_pred_scaled también lo está; lo invertimos
        return self.scaler.inverse_transform(combined_pred_scaled)


class ArquitecturaIACuanticaConRNN:
    """
    Integra la arquitectura cuántica multidimensional con nodos fotónicos y RNN.
    Esta clase combina la red de nodos fotónicos con un RNNCoordinator para
    procesamiento híbrido cuántico-clásico.
    """
    def __init__(self, dim_entrada=4, dim_nodos=3, dim_salida=2, 
                 hidden_size=64, rnn_type='GRU', dropout_rate=0.2, use_batch_norm=True):
        """
        Inicializa la arquitectura híbrida.
        
        Args:
            dim_entrada (int): Dimensión de los datos de entrada
            dim_nodos (int): Número de nodos fotónicos en la red
            dim_salida (int): Dimensión de salida del sistema
            hidden_size (int): Tamaño de la capa oculta de la RNN
            rnn_type (str): Tipo de RNN ('GRU' o 'LSTM')
            dropout_rate (float): Tasa de dropout para regularización
            use_batch_norm (bool): Usar normalización por lotes
        """
        self.dim_entrada = dim_entrada
        self.dim_nodos = dim_nodos
        self.dim_salida = dim_salida
        
        # Inicializar componentes
        self.red_nodos = RedNodosFotonicos(num_nodos=dim_nodos, dimension=dim_entrada)
        self.rnn_coordinator = RNNCoordinator(
            input_size=dim_entrada,
            hidden_size=hidden_size,
            output_size=dim_salida,
            rnn_type=rnn_type,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Estado cuántico global para integración bayesiana
        self.quantum_state = np.random.rand(2**4)  # 4 qubits = 16 estados
        self.quantum_state = self.quantum_state / np.sum(self.quantum_state)  # Normalizar
        
        # Historiales para seguimiento y análisis
        self.historial_coherencia = []
        self.historial_errores = []
        self.historial_bayesiano = {'alpha': 1.0, 'beta': 1.0}
        
    def procesar_datos(self, datos_entrada):
        """
        Procesa datos a través de la red de nodos fotónicos.
        
        Args:
            datos_entrada: Datos de entrada a procesar
            
        Returns:
            Tuple: (estados resultantes, características extraídas)
        """
        # Verificar que hay datos de entrada
        if len(datos_entrada) == 0:
            logger.warning("No hay datos de entrada para procesar.")
            return [], np.array([])
            
        # Normalizar entradas para procesamiento cuántico
        entradas_norm = []
        for x in datos_entrada:
            if np.linalg.norm(x) > 0:
                entradas_norm.append(x / np.linalg.norm(x))
            else:
                entradas_norm.append(x)
        
        # Propagar a través de la red de nodos fotónicos
        estados_nodos = self.red_nodos.propagar_informacion(entradas_norm)
        
        # Extraer características (probabilidades) de los estados cuánticos
        caracteristicas = np.array([np.abs(estado)**2 for estado in estados_nodos])
        
        # Guardar coherencia global
        coherencia = self.red_nodos.calcular_coherencia_global()
        self.historial_coherencia.append(coherencia)
        
        return estados_nodos, caracteristicas
    
    def entrenar_sistema(self, X_train, y_train, sequence_length=10, epochs=50, batch_size=32):
        """
        Entrena el sistema híbrido cuántico-RNN.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas
            sequence_length: Longitud de secuencia para RNN
            epochs: Épocas de entrenamiento
            batch_size: Tamaño de lote
            
        Returns:
            List: Historial de error
        """
        # Verificar que hay suficientes datos
        if len(X_train) <= sequence_length or len(y_train) <= sequence_length:
            logger.error("No hay suficientes datos para entrenar el sistema.")
            return self.historial_errores
            
        # Primero procesamos los datos con la red de nodos fotónicos
        logger.info("Procesando datos a través de la red de nodos fotónicos...")
        _, caracteristicas = self.procesar_datos(X_train)
        
        if len(caracteristicas) == 0:
            logger.warning("No se generaron características del procesamiento fotónico.")
            return self.historial_errores
            
        # Convertir características a formato adecuado para RNN
        X_procesado = np.vstack([c.flatten() for c in caracteristicas])
        
        # Asegurar que X_procesado tiene la misma cantidad de muestras que y_train
        if len(X_procesado) > len(y_train):
            X_procesado = X_procesado[:len(y_train)]
        elif len(X_procesado) < len(y_train):
            y_train = y_train[:len(X_procesado)]
        
        # Entrenar RNN con datos procesados
        logger.info("Entrenando RNN con datos procesados cuánticamente...")
        self.rnn_coordinator.train_models(
            X_procesado, y_train, 
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return self.historial_errores
    
    def predecir(self, X_test, sequence_length=10):
        """
        Realiza predicciones con el sistema híbrido.
        
        Args:
            X_test: Datos de prueba
            sequence_length: Longitud de secuencia para RNN
            
        Returns:
            Tuple[np.ndarray, float]: (Predicciones, confianza bayesiana)
        """
        # Verificar que hay suficientes datos
        if len(X_test) <= sequence_length:
            logger.warning("No hay suficientes datos para hacer predicciones.")
            return np.array([]), 0.5
            
        # Procesar datos con red de nodos fotónicos
        _, caracteristicas = self.procesar_datos(X_test)
        
        if len(caracteristicas) == 0:
            logger.warning("No se generaron características del procesamiento fotónico.")
            return np.array([]), 0.5
            
        # Convertir características a formato adecuado
        X_procesado = np.vstack([c.flatten() for c in caracteristicas])
        
        # Predicción con RNN
        predicciones = self.rnn_coordinator.predict(X_procesado, sequence_length)
        
        # Si no hay predicciones, devolver array vacío
        if len(predicciones) == 0:
            return predicciones, 0.5
            
        # Actualizar probabilidades bayesianas basadas en coherencia
        coherencia = self.historial_coherencia[-1] if self.historial_coherencia else 0.5
        if coherencia > 0.5:  # Si hay buena coherencia, confiamos más en el modelo
            self.historial_bayesiano['alpha'] += 1
        else:
            self.historial_bayesiano['beta'] += 1
            
        confianza_bayesiana = self.historial_bayesiano['alpha'] / (
            self.historial_bayesiano['alpha'] + self.historial_bayesiano['beta']
        )
        
        logger.info(f"Predicción completada. Confianza bayesiana: {confianza_bayesiana:.4f}")
        
        return predicciones, confianza_bayesiana
    
    def evaluar_modelo(self, X_test, y_test, sequence_length=10):
        """
        Evalúa el rendimiento del modelo híbrido.
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas reales
            sequence_length: Longitud de secuencia para RNN
            
        Returns:
            Dict: Métricas de evaluación
        """
        # Verificar que hay suficientes datos
        if len(X_test) <= sequence_length or len(y_test) <= sequence_length:
            logger.warning("No hay suficientes datos para evaluar el modelo.")
            return {
                'mse': float('nan'),
                'mae': float('nan'),
                'confianza_bayesiana': 0.5,
                'entropia_media': float('nan'),
                'coherencia_global': 0.0
            }
            
        # Hacer predicciones
        predicciones, confianza = self.predecir(X_test, sequence_length)
        
        if len(predicciones) == 0:
            logger.warning("No se generaron predicciones para evaluar.")
            return {
                'mse': float('nan'),
                'mae': float('nan'),
                'confianza_bayesiana': confianza,
                'entropia_media': float('nan'),
                'coherencia_global': self.historial_coherencia[-1] if self.historial_coherencia else 0
            }
            
        # Asegurar dimensiones compatibles
        min_len = min(len(predicciones), len(y_test))
        predicciones = predicciones[:min_len]
        y_test = y_test[:min_len]
        
        # Calcular métricas
        mse = np.mean((predicciones - y_test) ** 2)
        mae = np.mean(np.abs(predicciones - y_test))
        
        # Calcular entropía de Shannon de estados cuánticos
        if len(X_test) >= self.dim_nodos:
            estados_nodos, _ = self.procesar_datos(X_test[:self.dim_nodos])
            entropias = []
            for estado in estados_nodos:
                probs = np.abs(estado) ** 2
                entropia = StatisticalAnalysis.shannon_entropy(probs)
                entropias.append(entropia)
            
            entropia_media = np.mean(entropias) if entropias else float('nan')
        else:
            entropia_media = float('nan')
        
        resultados = {
            'mse': mse,
            'mae': mae,
            'confianza_bayesiana': confianza,
            'entropia_media': entropia_media,
            'coherencia_global': self.historial_coherencia[-1] if self.historial_coherencia else 0
        }
        
        return resultados
    
    def visualizar_resultados(self, X_test, y_test, sequence_length=10):
        """
        Genera visualizaciones de los resultados del modelo.
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas reales
            sequence_length: Longitud de secuencia para RNN
        """
        # Verificar que hay suficientes datos
        if len(X_test) <= sequence_length or len(y_test) <= sequence_length:
            logger.warning("No hay suficientes datos para visualizar resultados.")
            return
            
        # Obtener predicciones
        predicciones, _ = self.predecir(X_test, sequence_length)
        
        if len(predicciones) == 0:
            logger.warning("No se generaron predicciones para visualizar.")
            return
            
        # Asegurar dimensiones compatibles
        min_len = min(len(predicciones), len(y_test))
        predicciones = predicciones[:min_len]
        y_test = y_test[:min_len]
        
        # Verificar que hay al menos una dimensión en los datos
        if predicciones.shape[1] < 1 or y_test.shape[1] < 1:
            logger.warning("Las dimensiones de los datos no son suficientes para visualizar.")
            return
            
        # Crear figuras
        plt.figure(figsize=(15, 10))
        
        # 1. Gráfico de predicciones vs reales
        plt.subplot(2, 2, 1)
        plt.plot(y_test[:, 0], label='Real')
        plt.plot(predicciones[:, 0], label='Predicción')
        plt.title('Predicciones vs Valores Reales (primera dimensión)')
        plt.legend()
        
        # 2. Historial de coherencia cuántica
        plt.subplot(2, 2, 2)
        plt.plot(self.historial_coherencia)
        plt.title('Evolución de Coherencia Cuántica')
        plt.xlabel('Paso')
        plt.ylabel('Coherencia')
        
        # 3. Distribución de estados cuánticos para un nodo de ejemplo
        plt.subplot(2, 2, 3)
        if len(self.red_nodos.nodos) > 0:
            estado_ejemplo = self.red_nodos.nodos[0].estado
            probs = np.abs(estado_ejemplo) ** 2
            plt.bar(range(len(probs)), probs)
            plt.title(f'Distribución de Probabilidad - Nodo {0}')
            plt.xlabel('Estado Base')
            plt.ylabel('Probabilidad')
        
        # 4. Confianza bayesiana
        plt.subplot(2, 2, 4)
        alpha = self.historial_bayesiano['alpha']
        beta = self.historial_bayesiano['beta']
        x = np.linspace(0, 1, 100)
        y = beta_dist.pdf(x, alpha, beta)
        plt.plot(x, y)
        plt.title(f'Distribución Beta (α={alpha:.1f}, β={beta:.1f})')
        plt.xlabel('Confianza')
        plt.ylabel('Densidad')
        
        plt.tight_layout()
        plt.show()


def generar_datos_ejemplo(n_muestras=1000, dim=4, tipo='sinusoidal'):
    """
    Genera datos de ejemplo para probar el sistema.
    
    Args:
        n_muestras: Número de muestras
        dim: Dimensionalidad
        tipo: Tipo de datos ('sinusoidal', 'aleatorio', etc.)
    
    Returns:
        Tuple: (X, y) datos y etiquetas
    """
    if tipo == 'sinusoidal':
        X = np.zeros((n_muestras, dim))
        for i in range(dim):
            # Generamos señales sinusoidales con frecuencias diferentes
            X[:, i] = np.sin(np.linspace(0, 10 + i, n_muestras))
        
        # Las etiquetas son combinaciones de las señales
        y = np.zeros((n_muestras, 2))
        y[:, 0] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_muestras)
        y[:, 1] = 0.4 * X[:, 2] + 0.6 * X[:, 3] + 0.1 * np.random.randn(n_muestras)
        
    elif tipo == 'aleatorio':
        X = np.random.randn(n_muestras, dim)
        y = np.zeros((n_muestras, 2))
        for i in range(n_muestras):
            y[i, 0] = 0.5 * np.sum(X[i, :2]) + 0.1 * np.random.randn()
            y[i, 1] = 0.5 * np.sum(X[i, 2:]) + 0.1 * np.random.randn()
    
    else:
        raise ValueError(f"Tipo de datos '{tipo}' no reconocido")
    
    return X, y


def main():
    """Función principal de demostración"""
    # Configurar reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Generar datos de ejemplo
    logger.info("Generando datos de ejemplo...")
    X, y = generar_datos_ejemplo(n_muestras=1000, dim=4, tipo='sinusoidal')
    
    # Dividir en entrenamiento y prueba
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Crear y entrenar modelo
    logger.info("Creando modelo de IA Cuántica con RNN...")
    modelo = ArquitecturaIACuanticaConRNN(
        dim_entrada=4,
        dim_nodos=3,
        dim_salida=2,
        hidden_size=64,
        rnn_type='GRU',
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    logger.info("Entrenando modelo...")
    modelo.entrenar_sistema(X_train, y_train, sequence_length=10, epochs=20, batch_size=32)
    
    # Evaluar modelo
    logger.info("Evaluando modelo...")
    resultados = modelo.evaluar_modelo(X_test, y_test, sequence_length=10)
    logger.info(f"Resultados de evaluación: {resultados}")
    
    # Visualizar resultados
    logger.info("Visualizando resultados...")
    modelo.visualizar_resultados(X_test, y_test, sequence_length=10)
    
    return modelo


if __name__ == "__main__":
    main()
