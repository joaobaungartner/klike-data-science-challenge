"""Testes do Recommendations Engine a partir dos artefatos do repositório."""
from pathlib import Path
import numpy as np
import pytest
from src.recommendations_engine import RecommendationsEngine

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def engine():
    return RecommendationsEngine(ROOT)

@pytest.fixture
def campanha():
    return {
        "campaign_id":"DEMO-MANUAL-001", "platform":"Meta",
        "category":"E-commerce", "is_retargeting":False,
        "video_duration_s":16, "format":"vertical",
        "has_subtitle":False, "has_cta":False,
        "has_hook":False, "has_face":False, "text_density":"low"
    }

def test_entradas_invalidas(engine,campanha):
    for change in ({"platform":"inexistente"}, {"format":"circular"},
                   {"video_duration_s":-1}, {"has_face":2}):
        with pytest.raises(ValueError):
            engine.preparar_campanha({**campanha,**change})
    with pytest.raises(ValueError):
        engine.preparar_campanha({k:v for k,v in campanha.items() if k!="has_cta"})

def test_sem_leakage_das_metricas_posteriores(engine,campanha):
    x=engine.preparar_campanha(campanha)
    y=engine.preparar_campanha({**campanha,"ctr":.99,"roas":99,"klike_score":100})
    assert x.equals(y)
    assert np.allclose(engine.modelo_ridge.predict(x),engine.modelo_ridge.predict(y))

def test_treino_test_isolados(engine):
    assert len(engine.df_train)==400
    assert set(engine.df_train.index).isdisjoint(engine.test_indices)

def test_tres_recomendacoes_ordenadas(engine,campanha):
    result=engine.recomendar_campanha(campanha,n_bootstrap=2000)
    assert [r["feature"] for r in result["recomendacoes"]]==["has_hook","has_face","has_subtitle"]
    deltas=[r["delta_ridge"] for r in result["recomendacoes"]]
    assert len(deltas)==3 and deltas==sorted(deltas,reverse=True)
    assert all(r["n_original"]>=10 and r["n_proposto"]>=10 for r in result["recomendacoes"])

def test_reprodutibilidade(engine,campanha):
    a=engine.recomendar_campanha(campanha,n_bootstrap=2000)
    b=engine.recomendar_campanha(campanha,n_bootstrap=2000)
    assert a["total_recomendacoes"]==b["total_recomendacoes"]
    for x,y in zip(a["recomendacoes"],b["recomendacoes"]):
        assert x["feature"]==y["feature"]
        assert np.isclose(x["ic_inferior"],y["ic_inferior"])
        assert np.isclose(x["ic_superior"],y["ic_superior"])

@pytest.mark.parametrize("feature,original,proposto", [
    ("format", "vertical", "horizontal"),
    ("text_density", "low", "medium"),
    ("video_duration_s", 16, 20),
])
def test_descricao_categorica_e_numerica(engine, feature, original, proposto):
    descricao = engine.descrever_acao(feature, original, proposto)
    assert isinstance(descricao, str) and descricao


def test_recomendacoes_linha_real_dataset(engine):
    """Linha real do treino: exemplo com três recomendações."""
    campanha = engine.df_train.loc[
        engine.df_train["campaign_id"] == "KLK-0090"
    ].iloc[0]
    resultado = engine.recomendar_campanha(campanha, n_bootstrap=2000)
    assert resultado["campaign_id"] == "KLK-0090"
    assert resultado["total_recomendacoes"] == 3
    assert [x["feature"] for x in resultado["recomendacoes"]] == [
        "has_hook", "has_face", "has_subtitle"
    ]


def test_campanha_categorica_nao_quebra(engine):
    """Evita formatação numérica aplicada a características categóricas."""
    campanha = engine.df_train.loc[
        engine.df_train["campaign_id"] == "KLK-0169"
    ].iloc[0]
    resultado = engine.recomendar_campanha(campanha, n_bootstrap=2000)
    assert resultado["total_recomendacoes"] > 0
    assert any(x["feature"] == "text_density" for x in resultado["recomendacoes"])
