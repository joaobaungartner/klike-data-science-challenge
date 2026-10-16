# Klike Data Science Challenge

Projeto em Python com três etapas: análise exploratória (**Parte 1**), regressão do `klike_score` (**Parte 2**) e **Recommendations Engine híbrido e interativo (Parte 3)**.

## Teste o Recommendations Engine

1. Clone o repositório e abra a pasta raiz no VS Code.
2. Crie um ambiente virtual e instale as dependências:

   ```bash
   python -m venv .venv
   # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
   # Linux/macOS: source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. No VS Code, selecione o kernel desse ambiente e abra **[parte-3-recommendations-engine.ipynb](notebooks/parte-3-recommendations-engine.ipynb)**.
4. Execute todas as células uma vez, preencha os atributos de uma campanha e clique em **Gerar recomendações**. É possível testar novas campanhas repetidamente, sem editar código Python.

Também é possível executar no JupyterLab (`jupyter lab`). **O preview de notebooks no GitHub não executa `ipywidgets`**.

## Arquivos

| Caminho | Papel |
|---|---|
| `notebooks/parte-1.ipynb` | Análise exploratória e qualidade dos dados |
| `notebooks/parte-2.ipynb` | Modelagem e avaliação do Ridge reduzido |
| `notebooks/parte-3.ipynb` | Metodologia, análises de robustez, demonstração e testes |
| `notebooks/parte-3-recommendations-engine.ipynb` | **Formulário interativo**, entrada de campanhas e recomendações |
| `src/recommendations_engine.py` | Código reutilizável do motor de recomendação |
| `models/ridge_klike.joblib` | Pipeline treinado na Parte 2 |
| `models/features_klike.joblib` | Lista ordenada de features |
| `klike_challenge_dataset.csv` | Base original de 500 campanhas |

O dataset permanece na raiz para preservar os caminhos das Partes 1 e 2. Os notebooks aceitam execução iniciada na raiz ou dentro de `notebooks/`.

## Metodologia e cuidados

- O engine reutiliza o Ridge reduzido da Parte 2, **sem retreinamento**.
- Reconstrói a divisão original: 400 observações de treino e 100 de teste, com `random_state=42`.
- Todas as evidências e comparações contextuais vêm **somente do treino**, evitando usar o teste nas recomendações.
- Cria cenários modificando **uma característica criativa por vez** e estima a diferença prevista no Klike Score.
- Aplica comparações por plataforma e, quando há amostra suficiente, categoria; usa no mínimo dez registros por grupo.
- Considera consistência do CTR, intervalos percentílicos bootstrap, suporte dos grupos e regras de elegibilidade.
- Retorna **até três recomendações** com priorização pelo delta previsto. Se houver menos de três candidatas com suporte suficiente, explica a limitação em vez de inventar ganhos.

A configuração de exemplo do formulário (Meta / E-commerce, 16 segundos, sem hook, rosto, legendas e CTA, densidade low) permite demonstrar o funcionamento. Os valores são simulações e não representam ganhos comprovados.

### Limitações

O delta Ridge compara previsões de configurações alternativas, **não efeitos causais**. Diferenças entre médias podem ter confundimento; intervalos bootstrap não corrigem múltiplas comparações nem medem a incerteza do Ridge. A estrutura linear não modela interações explícitas por plataforma. As regras de no mínimo dez observações e delta mínimo de um ponto são heurísticas. As alterações são avaliadas isoladamente e os ganhos **não devem ser somados**.

## Reprodutibilidade

O pipeline `joblib` da Parte 2 foi serializado com **scikit-learn 1.9.1**. O `requirements.txt` fixa essa versão, pois versões diferentes podem falhar na desserialização ou alterar resultados. Carregue artefatos `joblib` somente de fontes confiáveis.

Para verificar a entrega: instale as dependências, selecione o kernel, reinicie o notebook e execute **Run All**. A documentação e as análises completas estão em `notebooks/parte-3.ipynb`.
