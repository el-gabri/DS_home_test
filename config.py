from pathlib import Path

# --- directory paths ---
PROJ_PATH_LIST = [r'C:\Users\gbhlisboa\DataspellProjects\Preco_Loja_Fisica']  # basta adicionar o seu caminho até o
# diretório raiz do  projeto

found_path = False
for path in PROJ_PATH_LIST:
    if Path(path).exists():
        PROJ_PATH = Path(path)
        found_path = True

if not found_path:
    PROJ_PATH = Path.cwd().parents[1]

DATA_PATH_RAW = PROJ_PATH / 'data' / 'raw'
DATA_PATH_WRANGLE = PROJ_PATH / 'data' / 'wrangle'
DATA_PATH_RESULTS = PROJ_PATH / 'data' / 'results'
DATA_PATH = PROJ_PATH / 'data'

# Multithread Colors
COR = ['BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE']
