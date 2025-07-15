import logging
import sys

def setup_logging():
    """
    Configura o sistema de logging para o projeto.

    Define o formato das mensagens de log, o nível de criticidade (INFO)
    e direciona a saída para o console (stdout).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout  # Envia logs para a saída padrão (console)
    )