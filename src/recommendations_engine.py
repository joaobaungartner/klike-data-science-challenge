"""Recommendations Engine híbrido: Ridge e evidências contextuais no treino.

As comparações observacionais e diferenças preditivas não são efeitos causais.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

class RecommendationsEngine:
    """Núcleo reutilizável de validação, simulação, bootstrap e priorização."""

    def __init__(self, project_root: str | Path, random_state: int = 42):
        root = Path(project_root).resolve()
        data_path = root / "data" / "klike_challenge_dataset.csv"
        model_path = root / "models" / "ridge_klike.joblib"
        features_path = root / "models" / "features_klike.joblib"
        for path in (data_path, model_path, features_path):
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo necessário não encontrado: {path}")

        self.RANDOM_STATE = random_state
        self.df = pd.read_csv(data_path)
        self.modelo_ridge = joblib.load(model_path)
        self.features_reduzidas = joblib.load(features_path)
        self.features_booleanas = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]
        self.features_categoricas = ["platform", "category", "format", "text_density"]
        self.features_contexto = ["platform", "category", "is_retargeting"]
        esperadas = ["platform", "category", "is_retargeting", "video_duration_s", "format", "has_subtitle", "has_cta", "has_hook", "has_face", "text_density"]
        if self.features_reduzidas != esperadas:
            raise ValueError("Features incompatíveis com o Ridge reduzido da Parte 2.")
        if not isinstance(self.modelo_ridge, Pipeline) or not isinstance(self.modelo_ridge.named_steps.get("model"), Ridge):
            raise TypeError("O artefato não contém o pipeline Ridge esperado.")
        if self.modelo_ridge.named_steps["model"].alpha != 1.0:
            raise ValueError("Alpha diferente da Parte 2.")
        if self.df["campaign_id"].duplicated().any() or self.df["klike_score"].isna().any():
            raise ValueError("Identificadores duplicados ou target ausente.")

        X = self.df[self.features_reduzidas].copy()
        for coluna in self.features_booleanas:
            X[coluna] = X[coluna].map({True: 1, False: 0}).astype(float)
        X_train, X_test, _, _ = train_test_split(X, self.df["klike_score"], test_size=.20, random_state=random_state)
        self.df_train = self.df.loc[X_train.index].copy()
        self.test_indices = frozenset(X_test.index)
        self.df_evidencias = self.df_train.copy()
        ctr_calculado = self.df_evidencias["clicks"] / self.df_evidencias["impressions"].replace(0, np.nan)
        valido = self.df_evidencias["ctr"].notna() & ctr_calculado.notna()
        inconsistente = valido & (self.df_evidencias["ctr"] - ctr_calculado).abs().gt(.0001)
        self.df_evidencias["ctr_consistente"] = self.df_evidencias["ctr"].mask(inconsistente)
        self.categorias_validas = {
            col: sorted(self.df_train[col].dropna().unique().tolist())
            for col in self.features_categoricas
        }
        if not set(self.df_train.index).isdisjoint(self.test_indices):
            raise AssertionError("Treino/teste não isolados.")

    def preparar_campanha(self, campanha):
        """
        Valida e prepara uma campanha para o Ridge reduzido.
    
        Parâmetros
        ----------
        campanha : dict ou pd.Series
            Características de uma campanha.
    
        Retorna
        -------
        pd.DataFrame
            DataFrame com uma linha e as 10 features
            na ordem esperada pelo modelo.
        """
    
        if isinstance(campanha, pd.Series):
            campanha = campanha.to_dict()
    
        if not isinstance(campanha, dict):
            raise TypeError(
                "A campanha deve ser um dict ou pd.Series."
            )
    
        features_ausentes = [
            feature
            for feature in self.features_reduzidas
            if feature not in campanha
        ]
    
        if features_ausentes:
            raise ValueError(
                f"Features ausentes: {features_ausentes}"
            )
    
        entrada = {
            feature: campanha[feature]
            for feature in self.features_reduzidas
        }
    
        # Validar variáveis categóricas
        for feature in self.features_categoricas:
            valor = entrada[feature]
    
            if pd.isna(valor) or valor not in self.categorias_validas[feature]:
                raise ValueError(
                    f"Valor inválido para {feature}: {valor}. "
                    f"Valores aceitos: {self.categorias_validas[feature]}"
                )
    
        # Converter booleanos para 0 ou 1
        for feature in self.features_booleanas:
            valor = entrada[feature]
    
            if pd.isna(valor):
                if feature == "has_subtitle":
                    entrada[feature] = np.nan
                    continue
    
                raise ValueError(
                    f"A feature {feature} não pode ser nula."
                )
    
            if isinstance(valor, (bool, np.bool_)):
                entrada[feature] = int(valor)
    
            elif isinstance(valor, (int, float, np.integer, np.floating)) \
                    and valor in (0, 1):
                entrada[feature] = int(valor)
    
            else:
                raise ValueError(
                    f"{feature} deve ser True, False, 0 ou 1."
                )
    
        # Validar a duração do vídeo
        duracao = entrada["video_duration_s"]
    
        if pd.isna(duracao):
            entrada["video_duration_s"] = np.nan
    
        elif (
            isinstance(duracao, (bool, np.bool_))
            or not isinstance(
                duracao,
                (int, float, np.integer, np.floating)
            )
            or not np.isfinite(duracao)
            or duracao <= 0
        ):
            raise ValueError(
                "video_duration_s deve ser um número positivo "
                "e finito, ou um valor ausente."
            )
    
        else:
            entrada["video_duration_s"] = float(duracao)
    
        return pd.DataFrame(
            [entrada],
            columns=self.features_reduzidas
        )


    def gerar_cenarios(self, campanha):
        """
        Gera cenários alternativos modificando uma feature
        por vez, mantendo o contexto original da campanha.
    
        Não calcula impacto nem classifica melhorias.
        """
    
        entrada = self.preparar_campanha(campanha)
        original = entrada.iloc[0].to_dict()
    
        cenarios = []
    
        def adicionar_cenario(feature, novo_valor, origem):
            alternativa = original.copy()
            alternativa[feature] = novo_valor
    
            cenarios.append({
                "feature_alterada": feature,
                "valor_original": original[feature],
                "valor_proposto": novo_valor,
                "origem": origem,
                **alternativa
            })
    
        # 1. Características booleanas
        booleanas_modificaveis = [
            "has_subtitle",
            "has_cta",
            "has_hook",
            "has_face"
        ]
    
        for feature in booleanas_modificaveis:
            valor = original[feature]
    
            if pd.notna(valor) and valor == 0:
                adicionar_cenario(
                    feature,
                    1,
                    "Inclusão de característica ausente"
                )
    
        # 2. Características categóricas
        categoricas_modificaveis = [
            "format",
            "text_density"
        ]
    
        for feature in categoricas_modificaveis:
            for valor in self.categorias_validas[feature]:
    
                if valor != original[feature]:
                    adicionar_cenario(
                        feature,
                        valor,
                        "Categoria observada no treinamento"
                    )
    
        # 3. Duração do vídeo
        duracao_atual = original["video_duration_s"]
    
        if pd.notna(duracao_atual):
    
            duracoes_plataforma = self.df_train.loc[
                self.df_train["platform"] == original["platform"],
                "video_duration_s"
            ].dropna()
    
            # Evitar referências baseadas em grupos muito pequenos
            if len(duracoes_plataforma) >= 20:
    
                quartis = duracoes_plataforma.quantile(
                    [0.25, 0.50, 0.75]
                )
    
                duracoes_candidatas = sorted({
                    int(round(valor))
                    for valor in quartis
                    if valor > 0
                })
    
                for duracao in duracoes_candidatas:
    
                    if not np.isclose(duracao, duracao_atual):
                        adicionar_cenario(
                            "video_duration_s",
                            duracao,
                            "Quartis da duração por plataforma no treino"
                        )
    
        # Organizar os cenários
        colunas = [
            "feature_alterada",
            "valor_original",
            "valor_proposto",
            "origem"
        ] + self.features_reduzidas
    
        return pd.DataFrame(cenarios, columns=colunas)


    def valores_iguais(self, a, b):
        if pd.isna(a) and pd.isna(b):
            return True
    
        return a == b


    def validar_cenarios(self, campanha, cenarios):
        original = self.preparar_campanha(campanha).iloc[0]
    
        for _, cenario in cenarios.iterrows():
    
            # Identificar todas as features alteradas
            alteradas = [
                feature
                for feature in self.features_reduzidas
                if not self.valores_iguais(
                    original[feature],
                    cenario[feature]
                )
            ]
    
            # Exatamente uma feature deve ser modificada
            assert len(alteradas) == 1, (
                f"Cenário inválido: {alteradas}"
            )
    
            assert alteradas[0] == cenario["feature_alterada"]
    
            # Preservar o contexto
            for feature in self.features_contexto:
                assert self.valores_iguais(
                    original[feature],
                    cenario[feature]
                )
    
            # Garantir que o cenário seja aceito pelo input
            self.preparar_campanha(
                cenario[self.features_reduzidas]
            )
    
        print("Todos os cenários passaram na validação.")
        print("Contexto preservado e apenas uma alteração por cenário.")


    def simular_impactos(self, campanha):
        """
        Calcula diferenças previstas no Klike Score
        entre uma campanha e seus cenários alternativos.
    
        Utiliza o Ridge reduzido já treinado,
        sem realizar novos ajustes.
        """
    
        # Preparar campanha original
        entrada_original = self.preparar_campanha(campanha)
    
        # Gerar alternativas válidas
        cenarios = self.gerar_cenarios(campanha)
    
        # Previsão da campanha original
        score_original = float(
            self.modelo_ridge.predict(entrada_original)[0]
        )
    
        colunas_resultado = [
            "feature_alterada",
            "valor_original",
            "valor_proposto",
            "origem",
            "score_previsto_original",
            "score_previsto_cenario",
            "delta_score_previsto"
        ]
    
        # Tratar campanhas sem alterações candidatas
        if cenarios.empty:
            return score_original, pd.DataFrame(
                columns=colunas_resultado
            )
    
        # Preparar alternativas para o mesmo pipeline
        X_cenarios = cenarios[self.features_reduzidas].copy()
    
        # Previsões sem novo treinamento
        scores_alternativos = self.modelo_ridge.predict(
            X_cenarios
        )
    
        # Organizar os resultados
        resultados = cenarios[
            [
                "feature_alterada",
                "valor_original",
                "valor_proposto",
                "origem"
            ]
        ].copy()
    
        resultados["score_previsto_original"] = score_original
        resultados["score_previsto_cenario"] = scores_alternativos
    
        resultados["delta_score_previsto"] = (
            resultados["score_previsto_cenario"]
            - resultados["score_previsto_original"]
        )
    
        # Ordenação exploratória pelo delta previsto
        resultados = resultados.sort_values(
            by="delta_score_previsto",
            ascending=False,
            kind="stable"
        ).reset_index(drop=True)
    
        return score_original, resultados


    def comparar_cenario_contextual(self, 
        campanha,
        feature,
        valor_proposto,
        minimo_por_grupo=10
    ):
        """
        Compara métricas observadas entre duas configurações
        de uma característica criativa.
    
        Utiliza somente self.df_evidencias (treinamento).
    
        A função não estima efeitos causais.
        """
    
        entrada = self.preparar_campanha(campanha).iloc[0]
    
        features_comparaveis = [
            "has_subtitle",
            "has_cta",
            "has_hook",
            "has_face",
            "format",
            "text_density"
        ]
    
        if feature not in features_comparaveis:
            raise ValueError(
                f"Feature não suportada nesta comparação: {feature}"
            )
    
        valor_original = entrada[feature]
    
        if pd.isna(valor_original):
            raise ValueError(
                "Não é possível comparar uma feature de valor desconhecido."
            )
    
        if feature in self.features_booleanas:
            if valor_proposto not in (0, 1, True, False):
                raise ValueError("Valor booleano proposto inválido.")
    
            valor_proposto = int(valor_proposto)
    
        elif valor_proposto not in self.categorias_validas[feature]:
            raise ValueError("Categoria proposta não observada no treino.")
    
        if valor_original == valor_proposto:
            raise ValueError(
                "A configuração proposta deve diferir da original."
            )
    
        metricas = {
            "klike_score": "Klike Score",
            "ctr_consistente": "CTR consistente",
            "engagement_rate": "Engagement Rate"
        }
    
        # Hierarquia de contextos, sem utilizar o conjunto de teste
        contextos = [
            (
                "Plataforma + categoria",
                self.df_evidencias.loc[
                    (self.df_evidencias["platform"] == entrada["platform"])
                    & (self.df_evidencias["category"] == entrada["category"])
                ]
            ),
            (
                "Plataforma",
                self.df_evidencias.loc[
                    self.df_evidencias["platform"] == entrada["platform"]
                ]
            )
        ]
    
        resultados = []
    
        for coluna_metrica, nome_metrica in metricas.items():
    
            comparacao = None
            ultimo_suporte = (0, 0)
    
            for nivel_contexto, dados_contexto in contextos:
    
                dados_validos = dados_contexto.loc[
                    dados_contexto[coluna_metrica].notna()
                    & dados_contexto[feature].notna()
                ]
    
                grupo_original = dados_validos.loc[
                    dados_validos[feature] == valor_original,
                    coluna_metrica
                ]
    
                grupo_proposto = dados_validos.loc[
                    dados_validos[feature] == valor_proposto,
                    coluna_metrica
                ]
    
                n_original = len(grupo_original)
                n_proposto = len(grupo_proposto)
    
                ultimo_suporte = (n_original, n_proposto)
    
                if (
                    n_original < minimo_por_grupo
                    or n_proposto < minimo_por_grupo
                ):
                    continue
    
                media_original = grupo_original.mean()
                media_proposta = grupo_proposto.mean()
    
                diferenca = media_proposta - media_original
    
                if coluna_metrica == "klike_score":
                    unidade = "pontos"
                    media_exibida_original = media_original
                    media_exibida_proposta = media_proposta
                    diferenca_exibida = diferenca
                    variacao_relativa = np.nan
    
                else:
                    unidade = "p.p."
                    media_exibida_original = media_original * 100
                    media_exibida_proposta = media_proposta * 100
                    diferenca_exibida = diferenca * 100
    
                    variacao_relativa = (
                        (diferenca / media_original) * 100
                        if media_original != 0
                        else np.nan
                    )
    
                comparacao = {
                    "Métrica": nome_metrica,
                    "Contexto": nivel_contexto,
                    "N original": n_original,
                    "N proposto": n_proposto,
                    "Média original": media_exibida_original,
                    "Média proposta": media_exibida_proposta,
                    "Diferença observada": diferenca_exibida,
                    "Unidade": unidade,
                    "Variação relativa (%)": variacao_relativa,
                    "Status": "Comparação disponível"
                }
    
                break
    
            if comparacao is None:
                comparacao = {
                    "Métrica": nome_metrica,
                    "Contexto": "Plataforma",
                    "N original": ultimo_suporte[0],
                    "N proposto": ultimo_suporte[1],
                    "Média original": np.nan,
                    "Média proposta": np.nan,
                    "Diferença observada": np.nan,
                    "Unidade": (
                        "pontos"
                        if coluna_metrica == "klike_score"
                        else "p.p."
                    ),
                    "Variação relativa (%)": np.nan,
                    "Status": "Amostra insuficiente"
                }
    
            resultados.append(comparacao)
    
        return pd.DataFrame(resultados)


    def integrar_evidencias_cenarios(self, 
        campanha,
        simulacoes,
        minimo_por_grupo=10
    ):
        """
        Integra previsões do Ridge e evidências contextuais.
    
        Retorna:
        - resumo: uma linha por cenário.
        - detalhes: uma linha por cenário e métrica avaliada.
    
        Não classifica causalidade nem gera recomendações.
        """
    
        colunas_metricas = {
            "Klike Score": "Delta Klike observado (pontos)",
            "CTR consistente": "Delta CTR observado (p.p.)",
            "Engagement Rate": "Delta Engagement observado (p.p.)"
        }
    
        resumos = []
        detalhes = []
    
        for id_cenario, (_, cenario) in enumerate(
            simulacoes.iterrows(), start=1
        ):
            feature = cenario["feature_alterada"]
    
            resumo = {
                "Cenario": id_cenario,
                "Feature": feature,
                "Original": cenario["valor_original"],
                "Proposto": cenario["valor_proposto"],
                "Delta Ridge (pontos)": float(
                    cenario["delta_score_previsto"]
                ),
                "Status evidencias": None
            }
    
            for coluna in colunas_metricas.values():
                resumo[coluna] = np.nan
    
            # A duração requer um método contextual específico
            if feature == "video_duration_s":
                resumo["Status evidencias"] = (
                    "Pendente: duracao numerica"
                )
                resumos.append(resumo)
                continue
    
            evidencias = self.comparar_cenario_contextual(
                campanha=campanha,
                feature=feature,
                valor_proposto=cenario["valor_proposto"],
                minimo_por_grupo=minimo_por_grupo
            )
    
            n_disponiveis = (
                evidencias["Status"] == "Comparação disponível"
            ).sum()
    
            if n_disponiveis == len(colunas_metricas):
                resumo["Status evidencias"] = "Completa"
            elif n_disponiveis > 0:
                resumo["Status evidencias"] = "Parcial"
            else:
                resumo["Status evidencias"] = "Insuficiente"
    
            for _, evidencia in evidencias.iterrows():
                metrica = evidencia["Métrica"]
                disponivel = (
                    evidencia["Status"] == "Comparação disponível"
                )
    
                if disponivel:
                    resumo[colunas_metricas[metrica]] = float(
                        evidencia["Diferença observada"]
                    )
    
                detalhes.append({
                    "Cenario": id_cenario,
                    "Feature": feature,
                    "Proposto": cenario["valor_proposto"],
                    "Metrica": metrica,
                    "Contexto": evidencia["Contexto"],
                    "N original": evidencia["N original"],
                    "N proposto": evidencia["N proposto"],
                    "Diferenca observada": evidencia[
                        "Diferença observada"
                    ],
                    "Unidade": evidencia["Unidade"],
                    "Variacao relativa (%)": evidencia[
                        "Variação relativa (%)"
                    ],
                    "Status": evidencia["Status"]
                })
    
            resumos.append(resumo)
    
        return pd.DataFrame(resumos), pd.DataFrame(detalhes)


    def comparar_duracao_contextual(self, 
        campanha,
        duracao_proposta,
        minimo_por_grupo=10
    ):
        """
        Compara faixas de duração utilizando os quartis
        da plataforma calculados somente no treinamento.
    
        As diferenças são observacionais, não causais.
        """
    
        entrada = self.preparar_campanha(campanha).iloc[0]
        duracao_original = entrada["video_duration_s"]
    
        if pd.isna(duracao_original):
            raise ValueError(
                "A duração original é desconhecida."
            )
    
        # Validar a duração alternativa
        alternativa = entrada.to_dict()
        alternativa["video_duration_s"] = duracao_proposta
    
        duracao_proposta = self.preparar_campanha(
            alternativa
        ).iloc[0]["video_duration_s"]
    
        if pd.isna(duracao_proposta):
            raise ValueError(
                "A duração proposta não pode ser nula."
            )
    
        if np.isclose(duracao_original, duracao_proposta):
            raise ValueError(
                "A duração proposta deve ser diferente da original."
            )
    
        # Quartis calculados apenas com dados de treinamento
        dados_plataforma = self.df_evidencias.loc[
            self.df_evidencias["platform"] == entrada["platform"]
        ]
    
        cortes = dados_plataforma[
            "video_duration_s"
        ].dropna().quantile([0.25, 0.50, 0.75]).to_numpy()
    
        if not np.all(np.isfinite(cortes)) or not np.all(
            np.diff(cortes) > 0
        ):
            raise ValueError(
                "Não foi possível construir quatro faixas distintas."
            )
    
        def identificar_faixa(duracao):
            return int(
                np.searchsorted(cortes, duracao, side="left") + 1
            )
    
        faixa_original = identificar_faixa(duracao_original)
        faixa_proposta = identificar_faixa(duracao_proposta)
    
        metricas = {
            "klike_score": "Klike Score",
            "ctr_consistente": "CTR consistente",
            "engagement_rate": "Engagement Rate"
        }
    
        contextos = [
            (
                "Plataforma + categoria",
                dados_plataforma.loc[
                    dados_plataforma["category"] == entrada["category"]
                ]
            ),
            (
                "Plataforma",
                dados_plataforma
            )
        ]
    
        resultados = []
    
        for coluna, nome_metrica in metricas.items():
    
            unidade = (
                "pontos"
                if coluna == "klike_score"
                else "p.p."
            )
    
            resultado = {
                "Métrica": nome_metrica,
                "Contexto": None,
                "Faixa original": faixa_original,
                "Faixa proposta": faixa_proposta,
                "N original": 0,
                "N proposto": 0,
                "Diferença observada": np.nan,
                "Unidade": unidade,
                "Variação relativa (%)": np.nan,
                "Status": "Amostra insuficiente"
            }
    
            if faixa_original == faixa_proposta:
                resultado["Status"] = "Mesma faixa de duração"
                resultados.append(resultado)
                continue
    
            for nivel, dados_contexto in contextos:
    
                dados_validos = dados_contexto.dropna(
                    subset=["video_duration_s", coluna]
                )
    
                faixas = np.searchsorted(
                    cortes,
                    dados_validos["video_duration_s"].to_numpy(),
                    side="left"
                ) + 1
    
                grupo_original = dados_validos.loc[
                    faixas == faixa_original, coluna
                ]
    
                grupo_proposto = dados_validos.loc[
                    faixas == faixa_proposta, coluna
                ]
    
                n_original = len(grupo_original)
                n_proposto = len(grupo_proposto)
    
                resultado.update({
                    "Contexto": nivel,
                    "N original": n_original,
                    "N proposto": n_proposto
                })
    
                if (
                    n_original < minimo_por_grupo
                    or n_proposto < minimo_por_grupo
                ):
                    continue
    
                media_original = grupo_original.mean()
                media_proposta = grupo_proposto.mean()
    
                diferenca = media_proposta - media_original
    
                if coluna != "klike_score":
                    diferenca *= 100
    
                variacao_relativa = (
                    100 * (
                        media_proposta - media_original
                    ) / media_original
                    if coluna != "klike_score"
                    and media_original != 0
                    else np.nan
                )
    
                resultado.update({
                    "Diferença observada": diferenca,
                    "Variação relativa (%)": variacao_relativa,
                    "Status": "Comparação disponível"
                })
    
                break
    
            resultados.append(resultado)
    
        return pd.DataFrame(resultados)


    def bootstrap_contextual(self, 
        campanha,
        feature,
        valor_proposto,
        metrica="klike_score",
        n_bootstrap=2000,
        random_state=42
    ):
        """
        Calcula intervalo percentílico bootstrap para
        a diferença observacional entre dois grupos.
    
        Preserva os contextos e as regras de comparação
        definidos nas funções anteriores.
        """
    
        nomes_metricas = {
            "klike_score": "Klike Score",
            "ctr_consistente": "CTR consistente",
            "engagement_rate": "Engagement Rate"
        }
    
        if metrica not in nomes_metricas:
            raise ValueError("Métrica não suportada.")
    
        entrada = self.preparar_campanha(campanha).iloc[0]
    
        # Recuperar a comparação previamente definida
        if feature == "video_duration_s":
            comparacao = self.comparar_duracao_contextual(
                campanha,
                valor_proposto
            )
        else:
            comparacao = self.comparar_cenario_contextual(
                campanha,
                feature,
                valor_proposto
            )
    
        info = comparacao.loc[
            comparacao["Métrica"] == nomes_metricas[metrica]
        ].iloc[0]
    
        if info["Status"] != "Comparação disponível":
            return {
                "Feature": feature,
                "Proposto": valor_proposto,
                "Métrica": nomes_metricas[metrica],
                "Status": info["Status"]
            }
    
        # Recuperar o contexto utilizado
        dados = self.df_evidencias.loc[
            self.df_evidencias["platform"] == entrada["platform"]
        ].copy()
    
        if info["Contexto"] == "Plataforma + categoria":
            dados = dados.loc[
                dados["category"] == entrada["category"]
            ].copy()
    
        dados = dados.dropna(subset=[feature, metrica])
    
        # Identificar os grupos de comparação
        if feature == "video_duration_s":
    
            referencia = self.df_evidencias.loc[
                self.df_evidencias["platform"] == entrada["platform"],
                "video_duration_s"
            ].dropna()
    
            cortes = referencia.quantile(
                [0.25, 0.50, 0.75]
            ).to_numpy()
    
            faixas = np.searchsorted(
                cortes,
                dados["video_duration_s"].to_numpy(),
                side="left"
            ) + 1
    
            grupo_original = dados.loc[
                faixas == info["Faixa original"],
                metrica
            ].to_numpy(dtype=float)
    
            grupo_proposto = dados.loc[
                faixas == info["Faixa proposta"],
                metrica
            ].to_numpy(dtype=float)
    
        else:
    
            grupo_original = dados.loc[
                dados[feature] == entrada[feature],
                metrica
            ].to_numpy(dtype=float)
    
            grupo_proposto = dados.loc[
                dados[feature] == valor_proposto,
                metrica
            ].to_numpy(dtype=float)
    
        # Verificar consistência com os grupos anteriores
        assert len(grupo_original) == info["N original"]
        assert len(grupo_proposto) == info["N proposto"]
    
        rng = np.random.default_rng(random_state)
    
        # Reamostragem independente dentro de cada grupo
        medias_originais = rng.choice(
            grupo_original,
            size=(n_bootstrap, len(grupo_original)),
            replace=True
        ).mean(axis=1)
    
        medias_propostas = rng.choice(
            grupo_proposto,
            size=(n_bootstrap, len(grupo_proposto)),
            replace=True
        ).mean(axis=1)
    
        diferencas = medias_propostas - medias_originais
    
        limite_inferior, limite_superior = np.quantile(
            diferencas,
            [0.025, 0.975]
        )
    
        diferenca_observada = (
            grupo_proposto.mean() - grupo_original.mean()
        )
    
        # Converter taxas em pontos percentuais
        fator = 1 if metrica == "klike_score" else 100
    
        return {
            "Feature": feature,
            "Proposto": valor_proposto,
            "Métrica": nomes_metricas[metrica],
            "Contexto": info["Contexto"],
            "N original": len(grupo_original),
            "N proposto": len(grupo_proposto),
            "Diferença observada": diferenca_observada * fator,
            "IC 95% inferior": limite_inferior * fator,
            "IC 95% superior": limite_superior * fator,
            "Inclui zero": limite_inferior <= 0 <= limite_superior,
            "Unidade": "pontos" if fator == 1 else "p.p.",
            "Status": "Bootstrap concluído"
        }


    def avaliar_robustez_cenarios(self, campanha, n_bootstrap=2000):
        """
        Avalia sistematicamente a robustez observacional
        dos cenários de uma campanha.
    
        Remove comparações redundantes de duração.
        Não seleciona recomendações finais.
        """
    
        _, simulacoes_campanha = self.simular_impactos(campanha)
    
        if simulacoes_campanha.empty:
            return pd.DataFrame(), pd.DataFrame()
    
        entrada = self.preparar_campanha(campanha).iloc[0]
    
        # Quartis da plataforma, calculados no treinamento
        duracoes = self.df_evidencias.loc[
            self.df_evidencias["platform"] == entrada["platform"],
            "video_duration_s"
        ].dropna()
    
        cortes = duracoes.quantile(
            [0.25, 0.50, 0.75]
        ).to_numpy()
    
        # Identificar cenários que representam a mesma comparação
        cenarios_unicos = []
        chaves_utilizadas = set()
    
        for _, cenario in simulacoes_campanha.iterrows():
    
            feature = cenario["feature_alterada"]
            valor = cenario["valor_proposto"]
    
            if feature == "video_duration_s":
                faixa = int(
                    np.searchsorted(cortes, valor, side="left") + 1
                )
                chave = (feature, faixa)
            else:
                chave = (feature, str(valor))
    
            if chave in chaves_utilizadas:
                continue
    
            chaves_utilizadas.add(chave)
            cenarios_unicos.append(cenario)
    
        metricas = [
            "klike_score",
            "ctr_consistente",
            "engagement_rate"
        ]
    
        detalhes = []
        resumos = []
    
        for cenario in cenarios_unicos:
    
            feature = cenario["feature_alterada"]
            valor = cenario["valor_proposto"]
    
            resultados_metrica = {}
    
            for metrica in metricas:
    
                resultado = self.bootstrap_contextual(
                    campanha=campanha,
                    feature=feature,
                    valor_proposto=valor,
                    metrica=metrica,
                    n_bootstrap=n_bootstrap,
                    random_state=self.RANDOM_STATE
                )
    
                detalhes.append(resultado)
                resultados_metrica[metrica] = resultado
    
            evidencia_score = resultados_metrica["klike_score"]
    
            if evidencia_score["Status"] != "Bootstrap concluído":
                classificacao = "Evidência insuficiente"
    
            else:
                inferior = evidencia_score["IC 95% inferior"]
                superior = evidencia_score["IC 95% superior"]
    
                if inferior > 0:
                    classificacao = "Associação positiva"
                elif superior < 0:
                    classificacao = "Associação negativa"
                else:
                    classificacao = "Inconclusiva"
    
            resumos.append({
                "Feature": feature,
                "Original": cenario["valor_original"],
                "Proposto": valor,
                "Delta Ridge": cenario["delta_score_previsto"],
                "Delta Klike observado": evidencia_score.get(
                    "Diferença observada", np.nan
                ),
                "IC inferior": evidencia_score.get(
                    "IC 95% inferior", np.nan
                ),
                "IC superior": evidencia_score.get(
                    "IC 95% superior", np.nan
                ),
                "Contexto": evidencia_score.get(
                    "Contexto", "Indisponível"
                ),
                "N original": evidencia_score.get(
                    "N original", np.nan
                ),
                "N proposto": evidencia_score.get(
                    "N proposto", np.nan
                ),
                "Classificação": classificacao
            })
    
        return (
            pd.DataFrame(resumos),
            pd.DataFrame(detalhes)
        )


    def priorizar_candidatos(self, 
        resumo_robustez,
        detalhes_robustez,
        delta_minimo=1.0,
        max_recomendacoes=3
    ):
        """
        Classifica cenários utilizando diferenças do Ridge
        e evidências observacionais.
    
        Retorna:
        - avaliacao: todos os candidatos classificados.
        - priorizados: até 3 candidatos elegíveis.
    
        Os critérios são exploratórios, não causais.
        """
    
        if delta_minimo < 0:
            raise ValueError("delta_minimo não pode ser negativo.")
    
        if max_recomendacoes < 1:
            raise ValueError("max_recomendacoes deve ser positivo.")
    
        avaliacao = resumo_robustez.copy()
        classificacoes = []
        justificativas = []
    
        for _, candidato in avaliacao.iterrows():
    
            delta = float(candidato["Delta Ridge"])
            associacao = candidato["Classificação"]
    
            # Localizar as evidências do cenário
            evidencias = detalhes_robustez.loc[
                (detalhes_robustez["Feature"] == candidato["Feature"])
                & (
                    detalhes_robustez["Proposto"]
                    == candidato["Proposto"]
                )
            ]
    
            # Verificar associações desfavoráveis em CTR/Engagement
            metricas_desfavoraveis = []
    
            for _, evidencia in evidencias.iterrows():
    
                if evidencia["Métrica"] not in [
                    "CTR consistente",
                    "Engagement Rate"
                ]:
                    continue
    
                if evidencia["Status"] != "Bootstrap concluído":
                    continue
    
                limite_superior = evidencia["IC 95% superior"]
    
                if pd.notna(limite_superior) and limite_superior < 0:
                    metricas_desfavoraveis.append(
                        evidencia["Métrica"]
                    )
    
            # Classificação dos candidatos
            if delta <= 0:
                status = "Não priorizar"
                motivo = "Diferença prevista não positiva."
    
            elif delta < delta_minimo:
                status = "Não priorizar"
                motivo = "Diferença prevista marginal."
    
            elif associacao == "Associação negativa":
                status = "Não priorizar"
                motivo = "Associação observacional desfavorável."
    
            elif metricas_desfavoraveis:
                status = "Revisão necessária"
                motivo = (
                    "Associação desfavorável em: "
                    + ", ".join(metricas_desfavoraveis)
                )
    
            elif associacao == "Associação positiva":
                status = "Elegível"
                motivo = (
                    "Delta previsto positivo e associação "
                    "observacional favorável."
                )
    
            elif associacao == "Inconclusiva":
                status = "Teste exploratório"
                motivo = (
                    "Delta previsto positivo, mas associação "
                    "observacional inconclusiva."
                )
    
            else:
                status = "Evidência insuficiente"
                motivo = "Suporte contextual insuficiente."
    
            classificacoes.append(status)
            justificativas.append(motivo)
    
        avaliacao["Status decisão"] = classificacoes
        avaliacao["Justificativa"] = justificativas
    
        # Ordenar todos os candidatos
        avaliacao = avaliacao.sort_values(
            "Delta Ridge",
            ascending=False,
            kind="stable"
        ).reset_index(drop=True)
    
        # Selecionar somente candidatos elegíveis
        priorizados = avaliacao.loc[
            avaliacao["Status decisão"] == "Elegível"
        ].copy()
    
        # Evitar múltiplas recomendações para a mesma feature
        priorizados = (
            priorizados
            .drop_duplicates(subset=["Feature"], keep="first")
            .head(max_recomendacoes)
            .reset_index(drop=True)
        )
    
        priorizados.insert(
            0,
            "Prioridade",
            range(1, len(priorizados) + 1)
        )
    
        return avaliacao, priorizados


    def descrever_acao(self, feature, original, proposto):
        """
        Converte a alteração de uma feature em uma
        sugestão compreensível para o time de marketing.
        """
    
        acoes_simples = {
            "has_hook": "Adicionar um hook nos primeiros 3 segundos",
            "has_face": "Incluir rosto humano no criativo",
            "has_subtitle": "Adicionar legendas ao vídeo",
            "has_cta": "Adicionar uma chamada para ação (CTA)",
        }
        if feature in acoes_simples:
            return acoes_simples[feature]
        if feature == "format":
            return f"Alterar o formato de {original} para {proposto}"
        if feature == "text_density":
            return f"Alterar a densidade de texto de {original} para {proposto}"
        if feature == "video_duration_s":
            return f"Alterar a duração de {float(original):g}s para {float(proposto):g}s"
        raise ValueError(f"Feature não suportada: {feature}")


    def recomendar_campanha(self, 
        campanha,
        max_recomendacoes=3,
        delta_minimo=1.0,
        n_bootstrap=2000
    ):
        """
        Recommendations Engine híbrido.
    
        Combina diferenças previstas pelo Ridge reduzido
        e associações contextuais obtidas no treinamento.
    
        Não estima efeitos causais.
        """
    
        # 1. Validar a entrada
        entrada = self.preparar_campanha(campanha)
    
        campanha_dict = (
            campanha.to_dict()
            if isinstance(campanha, pd.Series)
            else campanha
        )
    
        campaign_id = campanha_dict.get(
            "campaign_id",
            "Campanha sem identificador"
        )
    
        # 2. Calcular o score previsto da campanha original
        score_original = float(
            self.modelo_ridge.predict(entrada)[0]
        )
    
        # 3. Avaliar cenários e sua robustez
        resumo, detalhes = self.avaliar_robustez_cenarios(
            campanha=campanha,
            n_bootstrap=n_bootstrap
        )
    
        # 4. Tratar ausência de cenários
        if resumo.empty:
            return {
                "campaign_id": campaign_id,
                "score_original": score_original,
                "total_cenarios": 0,
                "total_recomendacoes": 0,
                "recomendacoes": [],
                "avaliacao": pd.DataFrame(),
                "evidencias": pd.DataFrame()
            }
    
        # 5. Aplicar os critérios de elegibilidade
        avaliacao, priorizados = self.priorizar_candidatos(
            resumo_robustez=resumo,
            detalhes_robustez=detalhes,
            delta_minimo=delta_minimo,
            max_recomendacoes=max_recomendacoes
        )
    
        recomendacoes = []
    
        # 6. Construir as recomendações
        for _, candidato in priorizados.iterrows():
    
            feature = candidato["Feature"]
            original = candidato["Original"]
            proposto = candidato["Proposto"]
    
            delta = float(candidato["Delta Ridge"])
    
            # Evidências da alteração selecionada
            evidencias_candidato = detalhes.loc[
                (detalhes["Feature"] == feature)
                & (detalhes["Proposto"] == proposto)
            ]
    
            evidencia_score = evidencias_candidato.loc[
                evidencias_candidato["Métrica"] == "Klike Score"
            ].iloc[0]
    
            # Construir a sugestão acionável
            acao = self.descrever_acao(
                feature,
                original,
                proposto
            )
    
            contexto = candidato["Contexto"]
    
            if contexto == "Plataforma + categoria":
                descricao_contexto = (
                    f"{entrada.iloc[0]['platform']} / "
                    f"{entrada.iloc[0]['category']}"
                )
            else:
                descricao_contexto = entrada.iloc[0]["platform"]
    
            diferenca_observada = float(
                evidencia_score["Diferença observada"]
            )
    
            limite_inferior = float(
                evidencia_score["IC 95% inferior"]
            )
    
            limite_superior = float(
                evidencia_score["IC 95% superior"]
            )
    
            texto = (
                f"{acao}. "
                f"O Ridge prevê uma diferença de "
                f"{delta:+.2f} pontos no Klike Score "
                f"para essa configuração alternativa. "
                f"Nas campanhas comparáveis de "
                f"{descricao_contexto}, a diferença média "
                f"observada entre os grupos foi de "
                f"{diferenca_observada:+.2f} pontos "
                f"(IC bootstrap 95%: "
                f"{limite_inferior:+.2f} a "
                f"{limite_superior:+.2f}). "
                f"Esses resultados não demonstram "
                f"uma relação causal."
            )
    
            recomendacoes.append({
                "prioridade": int(candidato["Prioridade"]),
                "feature": feature,
                "valor_original": original,
                "valor_proposto": proposto,
                "delta_ridge": delta,
                "contexto": descricao_contexto,
                "diferenca_observada": diferenca_observada,
                "ic_inferior": limite_inferior,
                "ic_superior": limite_superior,
                "n_original": int(evidencia_score["N original"]),
                "n_proposto": int(evidencia_score["N proposto"]),
                "texto": texto
            })
    
        # 7. Retornar resultado estruturado
        return {
            "campaign_id": campaign_id,
            "score_original": score_original,
            "total_cenarios": len(avaliacao),
            "total_recomendacoes": len(recomendacoes),
            "recomendacoes": recomendacoes,
            "avaliacao": avaliacao,
            "evidencias": detalhes
        }
