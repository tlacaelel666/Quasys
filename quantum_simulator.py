import logging
import math
import time
import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple

import numpy as np

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumHybridSystem")

# ===== SECCIÓN 1: CONSTANTES GLOBALES =====

EPSILON = 1e-10
MIN_LEARNING_RATE = 0.01
PROPAGATION_ITERATIONS = 3


# ===== SECCIÓN 2: CLASES DE ESTADO Y CONFIGURACIÓN =====

class EstadoQubit(Enum):
    """Estados posibles para un qubit superconductor"""
    GROUND = auto()  # |0⟩
    EXCITED = auto()  # |1⟩
    SUPERPOSITION = auto()  # α|0⟩ + β|1⟩
    UNKNOWN = auto()  # Estado desconocido o indeterminado


@dataclass
class EstadoComplejo:
    """Representación de un estado cuántico completo"""
    alpha: complex = complex(1.0)  # Amplitud del estado |0⟩
    beta: complex = complex(0.0)  # Amplitud del estado |1⟩

    def __post_init__(self):
        # Normalización automática
        self.normalize()

    def normalize(self):
        """Normaliza el vector de estado."""
        norm_sq = abs(self.alpha) ** 2 + abs(self.beta) ** 2
        if norm_sq > 1e-12:  # Evitar división por cero o normas muy pequeñas
            norm = np.sqrt(norm_sq)
            self.alpha /= norm
            self.beta /= norm
        else:
            # Si la norma es cero, resetear a |0⟩ por seguridad
            self.alpha = complex(1.0)
            self.beta = complex(0.0)

    @property
    def vector(self) -> np.ndarray:
        """Devuelve el vector de estado como array numpy"""
        return np.array([self.alpha, self.beta], dtype=complex)

    def probabilidad_0(self) -> float:
        """Probabilidad de medir |0⟩"""
        return abs(self.alpha) ** 2

    def probabilidad_1(self) -> float:
        """Probabilidad de medir |1⟩"""
        return abs(self.beta) ** 2

    def fase_relativa(self) -> float:
        """Calcula la fase relativa entre beta y alpha en radianes."""
        if abs(self.alpha) < 1e-9 or abs(self.beta) < 1e-9:
            return 0.0  # Fase no bien definida si una amplitud es cero
        return np.angle(self.beta) - np.angle(self.alpha)

    def __str__(self) -> str:
        return f"{self.alpha.real:+.4f}{self.alpha.imag:+.4f}j |0⟩ + {self.beta.real:+.4f}{self.beta.imag:+.4f}j |1⟩ (P0={self.probabilidad_0():.3f})"


@dataclass
class ConfiguracionNodoFotonico:
    """Configuración para nodos fotónicos"""
    dimension: int = 2
    tasa_aprendizaje: float = 0.1
    decay_adaptativo: float = 0.99
    prior_alpha_inicial: float = 1.0
    prior_beta_inicial: float = 1.0


@dataclass
class ConfiguracionRNN:
    """Configuración para RNN"""
    dimension_entrada: int
    dimension_oculta: int
    dimension_salida: int
    tipo_rnn: str = 'GRU'
    tasa_dropout: float = 0.2
    usar_batch_norm: bool = True


@dataclass
class MetricasSistema:
    """Métricas del sistema para la toma de decisiones"""
    ciclo: int
    tiempo_coherencia: float  # Tiempo de coherencia estimado del qubit en microsegundos
    temperatura: float  # Temperatura del sistema en milikelvin
    senal_ruido: float  # Relación señal-ruido (SNR) de lectura/control
    tasa_error: float  # Tasa de error de bit cuántico (QBER) estimada
    fotones_perdidos_acum: int  # Contador acumulado de fotones que no llegaron al destino
    calidad_transduccion: float  # Calidad estimada de la transducción (0 a 1)
    estado_enlace: Optional[Dict[str, Any]] = None  # Estado del enlace óptico
    voltajes_control: Optional[List[float]] = None  # Voltajes de control en varios puntos del sistema
    coherencia_red: float = 0.0  # Coherencia global de la red de nodos fotónicos

    def __str__(self) -> str:
        return (f"Métricas Ciclo {self.ciclo}: T_coh={self.tiempo_coherencia:.2f}μs, "
                f"Temp={self.temperatura:.2f}mK, SNR={self.senal_ruido:.2f}, "
                f"QBER={self.tasa_error:.4f}, Transd={self.calidad_transduccion:.2f}, "
                f"Coher={self.coherencia_red:.3f}, Fotones Perdidos={self.fotones_perdidos_acum}")


@dataclass
class EstadoFoton:
    """Estado simplificado de un fotón óptico para comunicación."""
    polarizacion: float  # Ángulo de polarización en radianes [0, pi]
    fase: float  # Fase relativa en radianes [0, 2*pi]
    valido: bool = True  # Indica si el fotón representa un estado válido

    def __str__(self) -> str:
        if not self.valido: return "Fotón Inválido/Perdido"
        return f"Fotón[pol={math.degrees(self.polarizacion):.1f}°, fase={math.degrees(self.fase):.1f}°]"


class OperacionCuantica(Enum):
    """Operaciones cuánticas disponibles"""
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    HADAMARD = auto()
    FASE_S = auto()
    RESET = auto()
    MEDICION = auto()


@dataclass
class ParametrosOperacion:
    """Parámetros para una operación cuántica"""
    tipo: OperacionCuantica
    angulo: Optional[float] = None  # Para rotaciones
    # Parámetros de pulso (opcionales)
    duracion_pulso: Optional[float] = None  # Duración en nanosegundos
    amplitud: Optional[float] = None  # Amplitud del pulso
    fase: Optional[float] = None  # Fase del pulso


@dataclass
class ConfiguracionSimulacion:
    """Configuración global de la simulación"""
    dimension: int = 2
    num_ciclos: int = 100
    intervalo_ciclo_s: float = 0.1
    num_nodos_fotonicos: int = 5
    longitud_canal_km: float = 0.5
    atenuacion_canal_db_km: float = 0.25
    eficiencia_transduccion: float = 0.8
    eficiencia_detector: float = 0.9
    temperatura_inicial_mK: float = 15.0
    t_coherencia_max_us: float = 100.0
    modo_debug: bool = False
    callbacks: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        if self.modo_debug:
            logger.setLevel(logging.DEBUG)


# ===== SECCIÓN 3: COMPONENTES CUÁNTICOS =====

class NodoFotonico:
    """Simula un nodo fotónico con procesamiento cuántico"""

    def __init__(self, id_nodo: int, config: ConfiguracionNodoFotonico):
        self.id = id_nodo
        self.config = config
        self.estado = self._inicializar_estado_cuantico()
        self.prior_alpha = config.prior_alpha_inicial
        self.prior_beta = config.prior_beta_inicial
        self.historial_coherencia = []  # Inicializar historial de coherencia
        self.tasa_aprendizaje = config.tasa_aprendizaje  # Inicializar tasa de aprendizaje

    def _inicializar_estado_cuantico(self) -> np.ndarray:
        """Inicializa el estado cuántico del nodo"""
        estado = np.random.rand(self.config.dimension) + 1j * np.random.rand(self.config.dimension)
        return estado / np.linalg.norm(estado)

    def actualizar_estado(self, estado_entrada: np.ndarray) -> np.ndarray:
        """Actualiza el estado del nodo basado en la entrada"""
        nuevo_estado = self.estado + 0.5 * estado_entrada
        self.estado = nuevo_estado / np.linalg.norm(nuevo_estado)

        coherencia = np.abs(np.mean(self.estado))
        self.historial_coherencia.append(coherencia)
        return self.estado

    def actualizar_tasa_aprendizaje(self) -> None:
        """Actualiza la tasa de aprendizaje adaptativamente"""
        self.tasa_aprendizaje *= self.config.decay_adaptativo
        self.tasa_aprendizaje = max(self.tasa_aprendizaje, MIN_LEARNING_RATE)


class RedNodosFotonicos:
    """Red de nodos fotónicos para procesamiento distribuido"""

    def __init__(self, num_nodos: int, config_nodo: ConfiguracionNodoFotonico):
        self.nodos = [NodoFotonico(i, config_nodo) for i in range(num_nodos)]
        self.matriz_conexiones = self._inicializar_matriz_conexiones(num_nodos)

    def _inicializar_matriz_conexiones(self, num_nodos: int) -> np.ndarray:
        """Inicializa la matriz de conexiones entre nodos"""
        matriz = np.random.rand(num_nodos, num_nodos)
        np.fill_diagonal(matriz, 0)
        return matriz

    def _inicializar_estados(self, datos_entrada: List[np.ndarray]) -> List[np.ndarray]:
        """Inicializa los estados de los nodos con datos de entrada"""
        estados = []
        for i, nodo in enumerate(self.nodos):
            if i < len(datos_entrada):
                # Si hay datos para este nodo, usarlos
                estado_normalizado = datos_entrada[i] / np.linalg.norm(datos_entrada[i])
                estados.append(estado_normalizado)
            else:
                # Si no hay datos, usar estado actual del nodo
                estados.append(nodo.estado)
        return estados

    def _calcular_estado_combinado(self, indice_nodo: int, estados_actuales: List[np.ndarray]) -> np.ndarray:
        """Calcula el estado combinado para un nodo basado en conexiones"""
        estado_combinado = np.zeros_like(estados_actuales[0], dtype=complex)

        for j, estado_j in enumerate(estados_actuales):
            if j != indice_nodo:  # No incluir el propio nodo
                peso = self.matriz_conexiones[indice_nodo, j]
                estado_combinado += peso * estado_j

        # Normalizar si no es cero
        norma = np.linalg.norm(estado_combinado)
        if norma > EPSILON:
            estado_combinado /= norma

        return estado_combinado

    def _actualizar_estado_nodo(self, nodo: NodoFotonico, estado_combinado: np.ndarray) -> np.ndarray:
        """Actualiza el estado de un nodo específico"""
        return nodo.actualizar_estado(estado_combinado)

    def _actualizar_estados_red(self, estados_actuales: List[np.ndarray]) -> List[np.ndarray]:
        """Actualiza los estados de todos los nodos en la red"""
        nuevos_estados = []
        for i, nodo in enumerate(self.nodos):
            estado_combinado = self._calcular_estado_combinado(i, estados_actuales)
            nuevos_estados.append(self._actualizar_estado_nodo(nodo, estado_combinado))
        return nuevos_estados

    def propagar_informacion(self, datos_entrada: List[np.ndarray]) -> List[np.ndarray]:
        """Propaga información a través de la red"""
        estados = self._inicializar_estados(datos_entrada)

        for _ in range(PROPAGATION_ITERATIONS):
            estados = self._actualizar_estados_red(estados)

        return estados

    def calcular_coherencia_global(self) -> float:
        """Calcula la coherencia global de la red"""
        coherencias = [nodo.historial_coherencia[-1] for nodo in self.nodos if nodo.historial_coherencia]
        return np.mean(coherencias) if coherencias else 0.0


class QubitSuperconductor:
    """Modelo simplificado de un qubit superconductor"""

    def __init__(self, id_qubit: str = "Q0", temp_inicial: float = 15.0, t_coherencia_max: float = 100.0):
        self.id = id_qubit
        self.estado_basico = EstadoQubit.GROUND
        self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
        self.tiempo_ultimo_reset = time.time()
        self.t_coherencia_max_base = t_coherencia_max  # microsegundos a temperatura base
        self._temperatura = temp_inicial  # milikelvin
        self.tiempo_coherencia_max = t_coherencia_max  # Inicializar tiempo de coherencia máximo
        self.actualizar_coherencia_por_temp()
        logger.info(
            f"Qubit {self.id} inicializado a |0⟩, Temp={self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    @property
    def temperatura(self) -> float:
        return self._temperatura

    @temperatura.setter
    def temperatura(self, valor: float):
        temp_anterior = self._temperatura
        self._temperatura = max(10.0, min(valor, 50.0))  # Limitada entre 10mK y 50mK
        if temp_anterior != self._temperatura:
            self.actualizar_coherencia_por_temp()
            logger.debug(
                f"Qubit {self.id} temp actualizada a {self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    def actualizar_coherencia_por_temp(self):
        """Ajusta el tiempo máximo de coherencia basado en la temperatura."""
        # Modelo simple: coherencia disminuye linealmente al aumentar temp sobre 10mK
        factor_temp = max(0, 1.0 - ((self._temperatura - 10.0) / 40.0) * 0.8)  # Pierde hasta 80% a 50mK
        self.tiempo_coherencia_max = self.t_coherencia_max_base * factor_temp

    def tiempo_desde_reset(self) -> float:
        """Tiempo transcurrido desde el último reset en segundos"""
        return time.time() - self.tiempo_ultimo_reset

    def aplicar_rotacion(self, eje: str, angulo: float):
        """Aplica una rotación en la esfera de Bloch."""
        cos_medio = math.cos(angulo / 2)
        sin_medio = math.sin(angulo / 2)

        if eje.upper() == 'X':
            matriz_rot = np.array([[cos_medio, -1j * sin_medio], [-1j * sin_medio, cos_medio]], dtype=complex)
        elif eje.upper() == 'Y':
            matriz_rot = np.array([[cos_medio, -sin_medio], [sin_medio, cos_medio]], dtype=complex)
        elif eje.upper() == 'Z':
            matriz_rot = np.array([[np.exp(-1j * angulo / 2), 0], [0, np.exp(1j * angulo / 2)]], dtype=complex)
        else:
            raise ValueError(f"Eje de rotación desconocido: {eje}")

        # Aplicar la matriz al estado
        vector_actual = self.estado_complejo.vector
        nuevo_vector = np.matmul(matriz_rot, vector_actual)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])

        # Actualizar estado básico basado en probabilidades (aproximado)
        prob_0 = self.estado_complejo.probabilidad_0()
        if abs(prob_0 - 1.0) < 0.01:  # Margen pequeño
            self.estado_basico = EstadoQubit.GROUND
        elif abs(prob_0 - 0.0) < 0.01:
            self.estado_basico = EstadoQubit.EXCITED
        else:
            self.estado_basico = EstadoQubit.SUPERPOSITION

    def aplicar_hadamard(self):
        """Aplica la compuerta Hadamard."""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        nuevo_vector = np.matmul(H, self.estado_complejo.vector)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])
        self.estado_basico = EstadoQubit.SUPERPOSITION  # Hadamard siempre crea superposición

    def aplicar_fase_s(self):
        """Aplica la compuerta de Fase S (sqrt(Z))."""
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        nuevo_vector = np.matmul(S, self.estado_complejo.vector)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])

    def reset(self):
        """Reinicia el qubit al estado base |0⟩"""
        self.estado_basico = EstadoQubit.GROUND
        self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
        self.tiempo_ultimo_reset = time.time()
        logger.info(f"Qubit {self.id} reiniciado a estado |0⟩")

    def simular_decoherencia(self):
        """Simula la decoherencia del qubit con el tiempo."""
        # Modelo T1 (relajación de amplitud) y T2 (decoherencia de fase)
        t = self.tiempo_desde_reset() * 1e6  # tiempo en microsegundos
        t1 = self.tiempo_coherencia_max * 1.5  # T1 suele ser mayor que T2
        t2 = self.tiempo_coherencia_max  # Usamos T_coh_max como T2

        if t > t2 / 10:  # Aplicar si ha pasado un tiempo significativo
            # Factor de decaimiento de fase (T2)
            factor_fase = np.exp(-t / t2)
            # Factor de decaimiento de amplitud (T1)
            factor_amp = np.exp(-t / t1)

            # Aplicar decaimiento de amplitud a beta (estado excitado)
            beta_amp = abs(self.estado_complejo.beta) * factor_amp
            # Recalcular alpha para mantener norma (aproximado, mejor usar matriz densidad)
            alpha_amp_sq = 1.0 - beta_amp ** 2
            alpha_amp = np.sqrt(max(0, alpha_amp_sq))

            # Aplicar decaimiento de fase a la fase relativa
            fase_rel_original = self.estado_complejo.fase_relativa()
            # La fase decae, pero modelarlo así es simplista. Mejor afectar coherencias off-diagonal.
            alpha_new = alpha_amp * (np.cos(np.angle(self.estado_complejo.alpha)) + 1j * np.sin(
                np.angle(self.estado_complejo.alpha)) * factor_fase)
            beta_new = beta_amp * (np.cos(np.angle(self.estado_complejo.beta)) + 1j * np.sin(
                np.angle(self.estado_complejo.beta)) * factor_fase)

            self.estado_complejo = EstadoComplejo(alpha_new, beta_new)

            # Si la decoherencia es muy alta, colapsar a estado clásico
            if t > t2:
                logger.warning(f"Qubit {self.id} ha decoherido significativamente (t={t:.1f}μs > T2={t2:.1f}μs)")
                self.medir()  # Forzar colapso por medición simulada
                self.estado_basico = EstadoQubit.GROUND if abs(
                    self.estado_complejo.alpha) > 0.5 else EstadoQubit.EXCITED

    def medir(self) -> int:
        """Simula una medición en la base computacional Z. Colapsa el estado."""
        prob_0 = self.estado_complejo.probabilidad_0()
        resultado = 0 if np.random.random() < prob_0 else 1

        # Colapsar estado
        if resultado == 0:
            self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
            self.estado_basico = EstadoQubit.GROUND
        else:
            self.estado_complejo = EstadoComplejo(complex(0.0), complex(1.0))
            self.estado_basico = EstadoQubit.EXCITED

        logger.info(f"Qubit {self.id} medido: Resultado = {resultado}. Estado colapsado a |{resultado}⟩")
        return resultado

    def __str__(self) -> str:
        return f"Qubit[{self.id}|{self.estado_basico.name}]: {self.estado_complejo}"


# ===== SECCIÓN 4: COMPONENTES ÓPTICOS Y CONTROL =====

class MicrowaveControl:
    """Controlador simulado de pulsos de microondas."""

    def __init__(self):
        self.frecuencia_base = 5.1  # GHz
        self.precision_tiempo = 0.1  # ns
        self.latencia_aplicacion = 5  # ns de retraso simulado
        # Calibración podría ser más compleja
        self.calibracion = {"offset_frecuencia": 0.0, "factor_amplitud": 1.0, "offset_fase": 0.0}
        logger.info("MicrowaveControl inicializado.")

    def traducir_operacion_a_pulso(self, operacion: ParametrosOperacion) -> Dict[str, Any]:
        """Traduce operación lógica a parámetros de pulso físico (simulado)."""
        # Parámetros base
        duracion_base_ns = 15.0
        amplitud_base = 0.95

        params = {
            "tipo_operacion": operacion.tipo.name,
            "angulo_logico": operacion.angulo,
            "duracion": duracion_base_ns,
            "amplitud": amplitud_base,
            "frecuencia": self.frecuencia_base + self.calibracion["offset_frecuencia"],
            "fase": self.calibracion["offset_fase"],
            "forma": "gaussiana_derivada"  # DRAG pulses etc.
        }

        # Ajustes específicos por operación
        if operacion.tipo == OperacionCuantica.ROTACION_X:
            # Rotación X requiere pulso en cuadratura (fase 0 o pi?)
            params["fase"] += 0.0
            # La duración/amplitud determina el ángulo de rotación
        elif operacion.tipo == OperacionCuantica.ROTACION_Y:
            params["fase"] += math.pi / 2
        elif operacion.tipo == OperacionCuantica.ROTACION_Z:
            # Virtual: se aplica cambiando marco de referencia (fase de futuros pulsos)
            params["virtual"] = True
            params["amplitud"] = 0.0  # No hay pulso físico
        elif operacion.tipo == OperacionCuantica.HADAMARD:
            # Secuencia específica o pulso calibrado
            params["tipo_operacion"] = "HADAMARD_PULSE"  # Pulso especial
            params["duracion"] = 25.0  # Hadamard suele ser más largo
        elif operacion.tipo == OperacionCuantica.FASE_S:
            params["virtual"] = True
            params["angulo_logico"] = math.pi / 2  # Rotación Z de 90 grados
            params["amplitud"] = 0.0

        params["amplitud"] *= self.calibracion["factor_amplitud"]

        return params

    def aplicar_pulso(self, qubit: QubitSuperconductor, params: Dict[str, Any]) -> str:
        """Simula la aplicación del pulso al qubit."""
        # Simular latencia
        time.sleep(self.latencia_aplicacion * 1e-9)

        # Simular decoherencia durante el pulso (simplificado)
        qubit.simular_decoherencia()  # Decoherencia natural justo antes
        qubit.estado_complejo.normalize()  # Renormalizar por si decoherencia afectó norma

        # Aplicar operación lógica
        tipo_op = params.get("tipo_operacion", "")
        resultado_op = "Éxito"
        angulo_op = params.get("angulo_logico")  # Ángulo deseado

        try:
            # Simular error de control (dependiente de amplitud, duración?)
            error_control_prob = 0.01 + (1.0 - params.get("amplitud", 1.0)) * 0.05
            if np.random.random() < error_control_prob:
                angulo_real = angulo_op * np.random.normal(1.0, 0.1) if angulo_op is not None else None
                logger.warning(
                    f"Error de control simulado! Ángulo aplicado: {angulo_real:.4f} vs deseado {angulo_op:.4f}")
            else:
                angulo_real = angulo_op

            if params.get("virtual", False):
                # Rotaciones virtuales Z o S
                if angulo_real is not None: qubit.aplicar_rotacion('Z', angulo_real)
                logger.debug(f"Pulso virtual {tipo_op} aplicado.")
            elif tipo_op == "ROTACION_X":
                if angulo_real is not None: qubit.aplicar_rotacion('X', angulo_real)
            elif tipo_op == "ROTACION_Y":
                if angulo_real is not None: qubit.aplicar_rotacion('Y', angulo_real)
            elif tipo_op == "HADAMARD_PULSE":
                # Simular Hadamard con posible error
                if np.random.random() > 0.02:  # 2% de fallo en Hadamard
                    qubit.aplicar_hadamard()
                else:
                    logger.error("Fallo simulado en compuerta Hadamard!")
                    resultado_op = "Fallo Hadamard"
            elif tipo_op == "RESET":
                qubit.reset()
            elif tipo_op == "MEDICION":
                med = qubit.medir()
                resultado_op = f"Medido |{med}⟩"
            else:
                logger.warning(f"Tipo de pulso/operación no manejado: {tipo_op}")
                resultado_op = "Operación no implementada"

            # Simular decoherencia post-pulso
            qubit.simular_decoherencia()

        except Exception as e:
            logger.error(f"Excepción al aplicar pulso {tipo_op}: {e}")
            resultado_op = "Error Excepción"

        return resultado_op


class TransductorSQaOptico:
    """Transductor simulado Superconductor -> Óptico."""

    def __init__(self, eficiencia_base: float = 0.8):
        self.eficiencia_conversion = eficiencia_base  # Probabilidad de éxito en transducción
        self.ruido_fase_polarizacion = 0.05  # Radianes de ruido añadido
        logger.info(f"Transductor SQ->Óptico inicializado. Eficiencia base: {self.eficiencia_conversion:.2f}")

    def leer_estado_sq(self, qubit: QubitSuperconductor) -> Optional[EstadoComplejo]:
        """Lee el estado del qubit (simulado, podría ser destructivo o QND)."""
        # Simular posible fallo en lectura basado en SNR (obtenido de métricas?)
        snr_simulado = 25.0  # Valor fijo para simplificar aquí
        prob_fallo_lectura = 0.01 + (1.0 - np.clip(snr_simulado / 30.0, 0, 1)) * 0.1
        if np.random.random() < prob_fallo_lectura:
            logger.warning(f"Fallo simulado en lectura de estado de {qubit.id}")
            return None  # Falla la lectura

        # Devolver copia del estado para no modificar el original si la lectura no es QND
        return EstadoComplejo(qubit.estado_complejo.alpha, qubit.estado_complejo.beta)


def mapear_estado_a_foton(self, estado_sq: EstadoComplejo) -> EstadoFoton:
    """Mapea el estado del qubit al estado de un fotón óptico."""
    # Validar entrada
    if estado_sq is None:
        return EstadoFoton(0.0, 0.0, valido=False)

    # Calcular polarización basada en las probabilidades
    # La polarización podría ser un mapeo del ángulo theta en la esfera de Bloch
    prob_0 = estado_sq.probabilidad_0()
    prob_1 = estado_sq.probabilidad_1()

    # Polarización (0 = horizontal, pi/2 = vertical) basada en probabilidades
    polarizacion = math.acos(math.sqrt(prob_0))

    # Fase del fotón basada en fase relativa del estado cuántico
    fase = estado_sq.fase_relativa()

    # Añadir ruido a la transducción
    polarizacion += np.random.normal(0, self.ruido_fase_polarizacion)
    fase += np.random.normal(0, self.ruido_fase_polarizacion)

    # Normalizar ángulos
    polarizacion = np.clip(polarizacion, 0, math.pi)
    fase = fase % (2 * math.pi)

    # Simular pérdida aleatoria basada en eficiencia
    if np.random.random() > self.eficiencia_conversion:
        logger.debug("Transducción falló - fotón no generado")
        return EstadoFoton(0.0, 0.0, valido=False)

    return EstadoFoton(polarizacion, fase, valido=True)


def actualizar_eficiencia(self, temperatura: float, calidad_substrato: float = 0.9):
    """Actualiza la eficiencia basada en condiciones físicas."""
    # La eficiencia empeora con temperatura más alta
    factor_temp = max(0.7, 1.0 - (temperatura - 10.0) / 100.0)
    self.eficiencia_conversion = min(0.95,
                                     0.8 * factor_temp * calidad_substrato)
