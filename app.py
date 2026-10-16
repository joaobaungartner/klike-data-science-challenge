"""Interface web do Klike. Inicie com: python -m streamlit run app.py."""
from pathlib import Path

import streamlit as st
from src.recommendations_engine import RecommendationsEngine

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Klike | Recommendations Engine", page_icon="🎯", layout="wide")

@st.cache_resource(show_spinner=False)
def carregar_engine():
    return RecommendationsEngine(ROOT)

try:
    engine = carregar_engine()
except (OSError, ValueError, KeyError, TypeError) as erro:
    st.error(f"Não foi possível carregar o modelo e os dados: {erro}")
    st.stop()

st.title("🎯 Klike Recommendations Engine")
st.write("Informe as características de um criativo e receba sugestões contextualizadas e priorizadas.")
st.caption("Protótipo demonstrativo. As estimativas não constituem efeitos causais.")

tab_nova, tab_dataset = st.tabs(["Nova campanha", "Campanha do dataset (treino)"])

with tab_nova:
    with st.form("campanha", clear_on_submit=False):
        st.subheader("1. Contexto da campanha")
        a, b, c = st.columns(3)
        with a:
            campaign_id = st.text_input("Identificador", value="CAMPANHA-NOVA")
            platform = st.selectbox("Plataforma", engine.categorias_validas["platform"], index=engine.categorias_validas["platform"].index("Meta"))
        with b:
            category = st.selectbox("Categoria", engine.categorias_validas["category"], index=engine.categorias_validas["category"].index("E-commerce"))
            video_duration_s = st.number_input("Duração (segundos)", min_value=1, max_value=10000, value=16, step=1)
        with c:
            video_format = st.selectbox("Formato", engine.categorias_validas["format"], index=engine.categorias_validas["format"].index("vertical"))
            text_density = st.selectbox("Densidade de texto", engine.categorias_validas["text_density"], index=engine.categorias_validas["text_density"].index("low"))
        st.subheader("2. Características do criativo")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            has_hook = st.checkbox("Hook (3 segundos)")
        with k2:
            has_face = st.checkbox("Rosto humano")
        with k3:
            has_subtitle = st.checkbox("Legendas")
        with k4:
            has_cta = st.checkbox("CTA")
        with k5:
            is_retargeting = st.checkbox("Retargeting")
        submitted = st.form_submit_button("Gerar recomendações", type="primary", width="stretch")
    

with tab_dataset:
    st.caption(
        "Exemplos limitados às 400 campanhas de treinamento. "
        "O conjunto de teste não é utilizado para selecionar ou justificar recomendações."
    )
    ids_treino = sorted(engine.df_train["campaign_id"].tolist())
    indice_padrao = ids_treino.index("KLK-0090") if "KLK-0090" in ids_treino else 0
    campanha_dataset_id = st.selectbox(
        "Campanha para avaliar", ids_treino, index=indice_padrao
    )
    linha_treino = engine.df_train.loc[
        engine.df_train["campaign_id"] == campanha_dataset_id
    ].iloc[0]
    st.dataframe(
        linha_treino[engine.features_reduzidas].to_frame("Valor").astype("string"),
        width="stretch"
    )
    submitted_dataset = st.button("Gerar recomendações do dataset", type="primary")

if submitted or submitted_dataset:
    if submitted_dataset:
        campanha = linha_treino.copy()
    else:
        campanha = {
            "campaign_id": campaign_id.strip() or "CAMPANHA-NOVA",
            "platform": platform, "category": category,
            "is_retargeting": is_retargeting, "video_duration_s": int(video_duration_s),
            "format": video_format, "has_subtitle": has_subtitle,
            "has_cta": has_cta, "has_hook": has_hook,
            "has_face": has_face, "text_density": text_density,
        }
    try:
        with st.spinner("Avaliando cenários e evidências contextuais..."):
            st.session_state["klike_resultado"] = engine.recomendar_campanha(
                campanha, max_recomendacoes=3, delta_minimo=1.0, n_bootstrap=2000
            )
    except (ValueError, TypeError, KeyError) as erro:
        st.session_state.pop("klike_resultado", None)
        st.error(f"Não foi possível avaliar a campanha: {erro}")

resultado = st.session_state.get("klike_resultado")
if resultado is not None:
    st.divider()
    st.subheader(f"Resultado · {resultado['campaign_id']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Klike Score previsto", f"{resultado['score_original']:.2f} pontos")
    m2.metric("Cenários avaliados", resultado["total_cenarios"])
    m3.metric("Recomendações elegíveis", resultado["total_recomendacoes"])

    st.subheader("Recomendações priorizadas")
    if resultado["recomendacoes"]:
        for rec in resultado["recomendacoes"]:
            with st.container(border=True):
                acao = engine.descrever_acao(
                    rec["feature"], rec["valor_original"], rec["valor_proposto"]
                )
                st.markdown(f"#### {rec['prioridade']}ª — {acao}")
                st.metric("Diferença prevista no Klike Score", f"{rec['delta_ridge']:+.2f} pontos")
                st.write(rec["texto"])
                st.caption(
                    f"Contexto: {rec['contexto']} · Grupos: "
                    f"{rec['n_original']} e {rec['n_proposto']} campanhas"
                )
    else:
        st.info("Nenhuma alteração atendeu aos critérios exploratórios de elegibilidade.")

    if resultado["total_recomendacoes"] < 3:
        st.info(
            "Esta campanha possui menos de três recomendações elegíveis. "
            "O sistema não inventa melhorias para completar a lista."
        )
    with st.expander("Ver diagnóstico completo"):
        if not resultado["avaliacao"].empty:
            colunas = [
                "Feature", "Original", "Proposto", "Delta Ridge",
                "Classificação", "Status decisão", "Justificativa",
            ]
            # Normaliza os campos mistos apenas para a exibição via PyArrow.
            tabela_diagnostico = resultado["avaliacao"][colunas].copy()
            for coluna in ("Original", "Proposto"):
                tabela_diagnostico[coluna] = tabela_diagnostico[coluna].astype("string")
            st.dataframe(tabela_diagnostico, width="stretch", hide_index=True)
        else:
            st.write("Não existem alternativas aplicáveis.")

    st.caption(
        "As diferenças do Ridge e as associações bootstrap não demonstram causalidade. "
        "Os critérios de elegibilidade são heurísticos."
    )

st.divider()
st.caption("Metodologia e validação: notebooks/parte-3.ipynb · Dataset simulado")
