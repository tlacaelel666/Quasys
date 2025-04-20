"""
Este archivo contiene la implementación de
la interfaz de línea de comandos (CLI) para
 el sistema de gestión de tareas Quasys.
fecha:20-04-2025
autor: jacobo tlacaelel mina rodriguez "jako".
version: Quasys 1.0
descripcion: Este script permite a los usuarios interactuar con el sistema

CLI para Quantum Simulator "Quasi" v1.0.0
Interfaz de línea de comandos para el Sistema Modular de Simulación Cuántica Híbrida
"""
# !/usr/bin/env python3

# python quasi_cli.py red medir
import argparse
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Importar el módulo principal del sistema
try:
    # Asumiendo que el código original está en un archivo llamado quantum_simulator.py
    from quantum_simulator import (
        ConfiguracionSimulacion, ConfiguracionNodoFotonico, ConfiguracionRNN,
        QubitSuperconductor, MicrowaveControl, TransductorSQaOptico,
        RedNodosFotonicos, EstadoQubit, EstadoComplejo, EstadoFoton,
        OperacionCuantica, ParametrosOperacion, MetricasSistema
    )
except ImportError:
    print("Error: No se pudo importar el módulo quantum_simulator.")
    print("Asegúrese de que el archivo quantum_simulator.py esté en el mismo directorio.")
    sys.exit(1)

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuasiCLI")


def mostrar_banner():
    """Muestra un banner ASCII art con el nombre Quosys"""
    banner = """
  ██████╗ ██╗   ██╗ ██████╗ ███████╗██╗   ██╗███████╗
 ██╔═══██╗██║   ██║██╔═══██╗██╔════╝╚██╗ ██╔╝██╔════╝
 ██║   ██║██║   ██║██║   ██║███████╗ ╚████╔╝ ███████╗
 ██║▄▄ ██║██║   ██║██║   ██║╚════██║  ╚██╔╝  ╚════██║
 ╚██████╔╝╚██████╔╝╚██████╔╝███████║   ██║   ███████║
  ╚══▀▀═╝  ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝

       Quantum Simulator System - v1.0.0
    """
    print(banner)


class QuasiCLI:
    """Interfaz de línea de comandos para Quantum Simulator Quasi"""

    def __init__(self):
        self.qubit = None
        self.microwave_control = None
        self.transductor = None
        self.red_fotonica = None
        self.config_simulacion = None
        self.metricas = None
        self.ciclo_actual = 1024

    def inicializar_sistema(self, config: Dict[str, Any]) -> None:
        """Inicializa los componentes del sistema con la configuración proporcionada"""
        logger.info("Inicializando sistema Quantum Simulator Quasi...")

        # Crear configuración de simulación
        self.config_simulacion = ConfiguracionSimulacion(
            num_ciclos=config.get('num_ciclos', 100),
            intervalo_ciclo_s=config.get('intervalo_ciclo_s', 0.1),
            num_nodos_fotonicos=config.get('num_nodos_fotonicos', 5),
            longitud_canal_km=config.get('longitud_canal_km', 0.5),
            atenuacion_canal_db_km=config.get('atenuacion_canal_db_km', 0.25),
            eficiencia_transduccion=config.get('eficiencia_transduccion', 0.8),
            eficiencia_detector=config.get('eficiencia_detector', 0.9),
            temperatura_inicial_mK=config.get('temperatura_inicial_mK', 15.0),
            t_coherencia_max_us=config.get('t_coherencia_max_us', 100.0),
            modo_debug=config.get('modo_debug', False)
        )

        # Inicializar componentes
        self.qubit = QubitSuperconductor(
            id_qubit="Q0",
            temp_inicial=self.config_simulacion.temperatura_inicial_mK,
            t_coherencia_max=self.config_simulacion.t_coherencia_max_us
        )

        self.microwave_control = MicrowaveControl()

        self.transductor = TransductorSQaOptico(
            eficiencia_base=self.config_simulacion.eficiencia_transduccion
        )

        # Configuración para nodos fotónicos
        config_nodo = ConfiguracionNodoFotonico(
            dimension=2,
            tasa_aprendizaje=0.1,
            decay_adaptativo=0.99,
            prior_alpha_inicial=1.0,
            prior_beta_inicial=1.0
        )

        # Inicializar red fotónica
        self.red_fotonica = RedNodosFotonicos(
            self.config_simulacion.num_nodos_fotonicos,
            config_nodo
        )

        # Inicializar métricas
        self.metricas = MetricasSistema(
            ciclo=0,
            tiempo_coherencia=self.config_simulacion.t_coherencia_max_us,
            temperatura=self.config_simulacion.temperatura_inicial_mK,
            senal_ruido=25.0,  # Valor inicial simulado
            tasa_error=0.01,  # Valor inicial simulado
            fotones_perdidos_acum=0,
            calidad_transduccion=self.config_simulacion.eficiencia_transduccion,
            coherencia_red=0.0
        )

        logger.info("Sistema inicializado correctamente.")
        print(
            f"Sistema Quosys v1.0.0 inicializado con {self.config_simulacion.num_nodos_fotonicos} nodos fotónicos.")

    def ejecutar_ciclo(self) -> MetricasSistema:
        """Ejecuta un ciclo completo de simulación"""
        self.ciclo_actual += 1
        logger.debug(f"Ejecutando ciclo {self.ciclo_actual}...")

        # 1. Simular decoherencia en qubit
        self.qubit.simular_decoherencia()

        # 2. Generar datos de entrada aleatorios para la red fotónica
        datos_entrada = [np.random.rand(2) + 1j * np.random.rand(2) for _ in range(2)]

        # 3. Propagar información en la red fotónica
        estados_salida = self.red_fotonica.propagar_informacion(datos_entrada)

        # 4. Actualizar métricas
        coherencia_red = self.red_fotonica.calcular_coherencia_global()

        # Actualizar objeto de métricas
        self.metricas = MetricasSistema(
            ciclo=self.ciclo_actual,
            tiempo_coherencia=self.qubit.tiempo_coherencia_max,
            temperatura=self.qubit.temperatura,
            senal_ruido=self.metricas.senal_ruido * 0.98 + 0.5,  # Simulación de cambio
            tasa_error=max(0.001, self.metricas.tasa_error * (1.0 + np.random.normal(0, 0.1))),
            fotones_perdidos_acum=self.metricas.fotones_perdidos_acum + (1 if np.random.random() > 0.8 else 0),
            calidad_transduccion=self.config_simulacion.eficiencia_transduccion * (0.95 + 0.1 * np.random.random()),
            coherencia_red=coherencia_red
        )

        # Simular pequeños cambios en la temperatura
        nuevo_temp = self.qubit.temperatura + np.random.normal(0, 0.1)
        self.qubit.temperatura = nuevo_temp

        return self.metricas

    def aplicar_operacion_qubit(self, operacion: OperacionCuantica, angulo: Optional[float] = None) -> str:
        """Aplica una operación al qubit y devuelve el resultado"""
        if self.qubit is None or self.microwave_control is None:
            return "Error: Sistema no inicializado"

        params = ParametrosOperacion(
            tipo=operacion,
            angulo=angulo
        )

        # Traducir operación a pulso
        params_pulso = self.microwave_control.traducir_operacion_a_pulso(params)

        # Aplicar pulso
        resultado = self.microwave_control.aplicar_pulso(self.qubit, params_pulso)

        return resultado

    def medir_qubit(self) -> int:
        """Mide el qubit y devuelve el resultado"""
        if self.qubit is None:
            logger.error("Error: Sistema no inicializado")
            return -1

        return self.qubit.medir()

    def transducir_a_foton(self) -> EstadoFoton:
        """Lee el estado del qubit y lo transduce a un fotón"""
        if self.qubit is None or self.transductor is None:
            logger.error("Error: Sistema no inicializado")
            return EstadoFoton(0.0, 0.0, False)

        estado_sq = self.transductor.leer_estado_sq(self.qubit)
        foton = self.transductor.mapear_estado_a_foton(estado_sq)

        return foton

    def mostrar_estado_actual(self) -> None:
        """Muestra el estado actual del sistema"""
        if self.qubit is None:
            print("Sistema no inicializado")
            return

        print("\n=== Estado Actual del Sistema ===")
        print(f"Ciclo: {self.ciclo_actual}")
        print(f"Qubit: {self.qubit}")
        print(f"Temperatura: {self.qubit.temperatura:.2f} mK")
        print(f"Tiempo de coherencia máx: {self.qubit.tiempo_coherencia_max:.2f} μs")
        print(f"Tiempo desde último reset: {self.qubit.tiempo_desde_reset():.2f} s")

        if self.metricas:
            print("\n=== Métricas ===")
            print(f"SNR: {self.metricas.senal_ruido:.2f}")
            print(f"QBER: {self.metricas.tasa_error:.6f}")
            print(f"Coherencia Red: {self.metricas.coherencia_red:.4f}")
            print(f"Fotones perdidos: {self.metricas.fotones_perdidos_acum}")
            print(f"Calidad transducción: {self.metricas.calidad_transduccion:.2f}")


def main():
    """Función principal de la CLI"""
    # Mostrar el banner al inicio
    mostrar_banner()

    parser = argparse.ArgumentParser(description="CLI para Quosys - Quantum Simulator System v1.0.0")

    # Comandos principales
    subparsers = parser.add_subparsers(dest="comando", help="Comandos disponibles")

    # Comando: inicializar
    parser_init = subparsers.add_parser("inicializar", help="Inicializar el sistema")
    parser_init.add_argument("--nodos", type=int, default=5, help="Número de nodos fotónicos")
    parser_init.add_argument("--temperatura", type=float, default=15.0, help="Temperatura inicial (mK)")
    parser_init.add_argument("--t-coherencia", type=float, default=100.0, help="Tiempo de coherencia máximo (μs)")
    parser_init.add_argument("--debug", action="store_true", help="Activar modo debug")

    # Comando: ciclo
    parser_ciclo = subparsers.add_parser("ciclo", help="Ejecutar un ciclo de simulación")
    parser_ciclo.add_argument("--n", type=int, default=1, help="Número de ciclos a ejecutar")

    # Comando: estado
    subparsers.add_parser("estado", help="Mostrar estado actual del sistema")

    # Comando: qubit
    parser_qubit = subparsers.add_parser("qubit", help="Operaciones sobre el qubit")
    parser_qubit.add_argument("operacion", choices=["rotX", "rotY", "rotZ", "H", "S", "reset", "medir"],
                              help="Operación a realizar")
    parser_qubit.add_argument("--angulo", type=float, help="Ángulo para rotaciones (radianes)")

    # Comando: transducir
    subparsers.add_parser("transducir", help="Transducir estado del qubit a fotón")

    # Comando: red
    parser_red = subparsers.add_parser("red", help="Operaciones sobre la red fotónica")
    parser_red.add_argument("operacion", choices=["propagar", "coherencia"], help="Operación a realizar")

    # Analizamos los argumentos
    args = parser.parse_args()

    # Instancia de la CLI
    cli = QuasiCLI()

    # Ejecutar el comando correspondiente
    if args.comando == "inicializar":
        config = {
            'num_nodos_fotonicos': args.nodos,
            'temperatura_inicial_mK': args.temperatura,
            't_coherencia_max_us': args.t_coherencia,
            'modo_debug': args.debug
        }
        cli.inicializar_sistema(config)

    elif args.comando == "ciclo":
        if cli.qubit is None:
            print("Error: Debe inicializar el sistema primero con 'inicializar'")
            return

        print(f"Ejecutando {args.n} ciclo(s) de simulación...")
        for i in range(args.n):
            metricas = cli.ejecutar_ciclo()
            if i == args.n - 1 or args.n == 1:  # Mostrar solo el último ciclo o si solo es uno
                print(f"Ciclo {cli.ciclo_actual}: T={metricas.temperatura:.2f}mK, "
                      f"Coherencia={metricas.coherencia_red:.4f}, QBER={metricas.tasa_error:.6f}")

    elif args.comando == "estado":
        cli.mostrar_estado_actual()

    elif args.comando == "qubit":
        if cli.qubit is None:
            print("Error: Debe inicializar el sistema primero con 'inicializar'")
            return

        # Mapear operación de string a enum OperacionCuantica
        operacion_map = {
            "rotX": OperacionCuantica.ROTACION_X,
            "rotY": OperacionCuantica.ROTACION_Y,
            "rotZ": OperacionCuantica.ROTACION_Z,
            "H": OperacionCuantica.HADAMARD,
            "S": OperacionCuantica.FASE_S,
            "reset": OperacionCuantica.RESET,
            "medir": OperacionCuantica.MEDICION
        }

        op = operacion_map[args.operacion]

        # Para rotaciones necesitamos ángulo
        if args.operacion in ["rotX", "rotY", "rotZ"] and args.angulo is None:
            print(f"Error: La operación {args.operacion} requiere un ángulo (--angulo)")
            return

        resultado = cli.aplicar_operacion_qubit(op, args.angulo)
        print(f"Operación {args.operacion} aplicada. Resultado: {resultado}")
        print(f"Estado actual: {cli.qubit}")

    elif args.comando == "transducir":
        if cli.qubit is None:
            print("Error: Debe inicializar el sistema primero con 'inicializar'")
            return

        foton = cli.transducir_a_foton()
        if foton.valido:
            print(f"Transducción exitosa: {foton}")
            print(f"  - Polarización: {np.degrees(foton.polarizacion):.2f}°")
            print(f"  - Fase: {np.degrees(foton.fase):.2f}°")
        else:
            print("Transducción fallida: Fotón inválido/perdido")

    elif args.comando == "red":
        if cli.red_fotonica is None:
            print("Error: Debe inicializar el sistema primero con 'inicializar'")
            return

        if args.operacion == "coherencia":
            coherencia = cli.red_fotonica.calcular_coherencia_global()
            print(f"Coherencia global de la red: {coherencia:.6f}")
        elif args.operacion == "propagar":
            datos_entrada = [np.random.rand(2) + 1j * np.random.rand(2) for _ in range(2)]
            estados_salida = cli.red_fotonica.propagar_informacion(datos_entrada)
            print(f"Propagación completada. Estados resultantes en {len(estados_salida)} nodos.")

    else:
        print("Quosys - Quantum Simulator System v1.0.0 CLI")
        print("Uso: python quasi_cli.py [comando] [opciones]")
        print("Ejecute 'python quasi_cli.py -h' para ver la ayuda.")


if __name__ == "__main__":
    main()

    # ejemplo de uso:
    # quasi_cli.py
    # Inicializar el sistema
    # python quasi_cli.py inicializar --nodos 5 --temperatura 15.0 --t-coherencia 100.0
    # python quasi_cli.py inicializar
    # Ver el estado actual
    # python quasi_cli.py estado

    # Ejecutar un ciclo de simulación
    # python quasi_cli.py ciclo --n 3

    # Aplicar operaciones al qubit
    # python quasi_cli.py qubit H
    # python quasi_cli.py qubit rotX --angulo 1.57
    # python quasi_cli.py qubit medir

    # Transducir el estado a un fotón
    # python quasi_cli.py transducir

    # Operaciones sobre la red fotónica
    # python quasi_cli.py red coherencia