# main.py
import logging
import sys
from asyncio.log import logger

from quasi_cli import QuasiCLI
from quantum_simulator import ConfiguracionSimulacion, QubitSuperconductor, RedNodosFotonicos

def configurar_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Main")
    return logger

def inicializar_simulacion(config):
    """Inicializa los componentes principales de la simulación."""
    logger.info("Inicializando simulación...")
    qubit = QubitSuperconductor(
        id_qubit="Q0",
        temp_inicial=config.temperatura_inicial_mK,
        t_coherencia_max=config.t_coherencia_max_us
    )
    red = RedNodosFotonicos(
        num_nodos=config.num_nodos_fotonicos,
        config_nodo=config.t_coherencia_max_us
    )
    return qubit, red

def main():
    """Punto de entrada principal del programa."""
    logger = configurar_logging()
    logger.info("Iniciando Quantum Simulator Quasi v1.0.0")

    try:
        # Configuración inicial
        config = ConfiguracionSimulacion(
            num_ciclos=100,
            intervalo_ciclo_s=0.1,
            num_nodos_fotonicos=5,
            temperatura_inicial_mK=15.0,
            t_coherencia_max_us=100.0,
            modo_debug=False
        )

        # Inicializar CLI
        cli = QuasiCLI()

        # Ejecutar CLI
        # python quasi_cli

    except Exception as e:
        logger.error(f"Error crítico: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()


