"""Prepare a Parte 1 para executar a partir de notebooks/ ou da raiz.

Necessário porque o notebook original usa o caminho legado do CSV.
Apenas o código da célula de carregamento é ajustado; resultados não são alterados.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "parte-1.ipynb"

def main():
    notebook = json.loads(PATH.read_text(encoding="utf-8"))
    old = 'df_raw = pd.read_csv("klike_challenge_dataset.csv")'
    replacement = (
        'from pathlib import Path\n'
        'PROJECT_ROOT = Path.cwd().resolve()\n'
        'if not (PROJECT_ROOT / "data" / "klike_challenge_dataset.csv").is_file():\n'
        '    PROJECT_ROOT = PROJECT_ROOT.parent\n'
        'DATA_PATH = PROJECT_ROOT / "data" / "klike_challenge_dataset.csv"\n'
        'if not DATA_PATH.is_file():\n'
        '    raise FileNotFoundError("Dataset não encontrado em data/.")\n'
        '\n'
        'df_raw = pd.read_csv(DATA_PATH)'
    )
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell.get("source", []))
        if old in text:
            cell["source"] = (text.replace(old, replacement)).splitlines(keepends=True)
            changed = True
            break
    if changed:
        PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Corrigido: {PATH}")
    else:
        print("O notebook já foi corrigido, ou sua estrutura não corresponde à esperada.")

if __name__ == "__main__":
    main()
