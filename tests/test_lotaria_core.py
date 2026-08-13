"""
Testes básicos de lotaria_core.py.

Corro com: pytest tests/
Cada teste usa uma pasta temporária para eu não poluir o disco do projecto.
"""

import os

import pandas as pd
import pytest

from lotaria_core import (
    Config,
    RepositorioResultados,
    RepositorioBase,
    ResultadoInvalidoError,
    AnalisadorLotaria,
    calcular_features,
    criar_repositorio,
    frequencias_por_numero,
    atrasos_por_numero,
)


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        arquivo_csv=str(tmp_path / "resultados.csv"),
        pasta_backup=str(tmp_path / "backups"),
        minimo_sorteios_para_treino=8,
    )


def test_cria_dados_de_exemplo_quando_csv_nao_existe(config):
    repo = RepositorioResultados(config)
    assert not repo.df.empty
    assert os.path.exists(config.arquivo_csv)


def test_adicionar_resultado_valido(config):
    repo = RepositorioResultados(config)
    total_antes = len(repo.df)
    repo.adicionar_resultado("2026-08-09", "Fezada", [1, 2, 3, 4, 5])
    assert len(repo.df) == total_antes + 1


def test_adicionar_resultado_substitui_duplicado_mesma_data_sorteio(config):
    repo = RepositorioResultados(config)
    repo.adicionar_resultado("2026-08-09", "Fezada", [1, 2, 3, 4, 5])
    total_apos_primeiro = len(repo.df)
    repo.adicionar_resultado("2026-08-09", "Fezada", [6, 7, 8, 9, 10])
    assert len(repo.df) == total_apos_primeiro  # substituiu, nao duplicou


@pytest.mark.parametrize(
    "sorteio,numeros,motivo",
    [
        ("SorteioInexistente", [1, 2, 3, 4, 5], "sorteio invalido"),
        ("Fezada", [1, 2, 3, 4], "quantidade errada"),
        ("Fezada", [1, 1, 2, 3, 4], "numeros repetidos"),
        ("Fezada", [0, 2, 3, 4, 5], "abaixo do minimo"),
        ("Fezada", [1, 2, 3, 4, 91], "acima do maximo"),
    ],
)
def test_adicionar_resultado_invalido_gera_erro(config, sorteio, numeros, motivo):
    repo = RepositorioResultados(config)
    with pytest.raises(ResultadoInvalidoError):
        repo.adicionar_resultado("2026-08-09", sorteio, numeros)


def test_calcular_features_com_historico_vazio_retorna_zeros():
    df_vazio = pd.DataFrame(columns=["N1", "N2", "N3", "N4", "N5"])
    features = calcular_features(df_vazio, numero=7)
    assert (features == 0).all()


def test_calcular_features_numero_impar_ultima_posicao():
    df = pd.DataFrame([{"N1": 1, "N2": 2, "N3": 3, "N4": 4, "N5": 7}])
    features = calcular_features(df, numero=7)
    assert features[-1] == 0.0  # 7 e impar


def test_previsao_geral_retorna_quantidade_pedida(config):
    analyzer = AnalisadorLotaria(config)
    previsao, scores = analyzer.prever_numeros(qtd=5)
    assert len(previsao) == 5
    assert len(scores) == 90
    assert previsao == sorted(previsao)


class RepositorioFake:
    """Repositório em memória: usei-o só para provar que o AnalisadorLotaria
    funciona com qualquer implementacao de RepositorioBase (Protocol),
    sem precisar de um Supabase real rodando durante os testes."""

    def __init__(self, dados: list[tuple]):
        self._df = pd.DataFrame(dados, columns=["Data", "Sorteio", "N1", "N2", "N3", "N4", "N5"])
        self._df["Data"] = pd.to_datetime(self._df["Data"])

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def adicionar_resultado(self, data, sorteio, numeros) -> None:
        RepositorioResultados.validar_resultado(sorteio, numeros)
        nova = pd.DataFrame([{
            "Data": pd.to_datetime(data), "Sorteio": sorteio,
            **dict(zip(["N1", "N2", "N3", "N4", "N5"], numeros)),
        }])
        self._df = pd.concat([self._df, nova], ignore_index=True)


def test_repositorio_fake_cumpre_o_protocolo():
    fake = RepositorioFake([("2026-08-01", "Fezada", 1, 2, 3, 4, 5)])
    assert isinstance(fake, RepositorioBase)


def test_analisador_funciona_com_repositorio_injetado():
    dados = [
        ("2026-08-0" + str(i % 9 + 1), "Fezada", i, i + 1, i + 2, i + 3, i + 4)
        for i in range(1, 10)
    ]
    # garante numeros unicos e dentro do intervalo em cada linha
    dados = [
        ("2026-08-01", "Fezada", 1, 2, 3, 4, 5),
        ("2026-08-02", "Fezada", 6, 7, 8, 9, 10),
        ("2026-08-03", "Fezada", 11, 12, 13, 14, 15),
    ]
    analyzer = AnalisadorLotaria(repositorio=RepositorioFake(dados))
    previsao, _ = analyzer.prever_numeros(qtd=5)
    assert len(previsao) == 5


def test_criar_repositorio_supabase_sem_credenciais_da_erro_claro():
    with pytest.raises(ValueError, match="SUPABASE_URL"):
        criar_repositorio(backend="supabase")


def test_criar_repositorio_csv_e_o_padrao(config):
    repo = criar_repositorio(config, backend="csv")
    assert isinstance(repo, RepositorioResultados)


def test_previsao_sem_historico_suficiente_nao_quebra(tmp_path):
    """Com poucos sorteios, espero que o treino seja saltado (retorna None)
    e a previsao deve cair para o modo so-estatistico, sem erro."""
    config = Config(
        arquivo_csv=str(tmp_path / "resultados.csv"),
        pasta_backup=str(tmp_path / "backups"),
        minimo_sorteios_para_treino=999,
    )
    analyzer = AnalisadorLotaria(config)
    previsao, _ = analyzer.prever_numeros(qtd=5)
    assert len(previsao) == 5


def test_frequencias_por_numero_cobre_1_a_90(config):
    analyzer = AnalisadorLotaria(config)
    freq = frequencias_por_numero(analyzer.df)
    assert list(freq.index) == list(range(1, 91))
    assert freq.sum() == len(analyzer.df) * 5  # 5 numeros por sorteio


def test_frequencias_por_numero_bate_com_calcular_features(config):
    """Garanti que o atalho vectorizado (usado na Grelha) concorda com
    o calculo por numero (usado na previsao), evitando divergencia
    entre as duas formas de calcular frequencia."""
    analyzer = AnalisadorLotaria(config)
    freq_bulk = frequencias_por_numero(analyzer.df)
    for numero in [1, 15, 47, 90]:
        freq_individual = calcular_features(analyzer.df, numero)[0] * len(analyzer.df)
        assert abs(freq_bulk[numero] - freq_individual) < 1e-6


def test_atrasos_por_numero_com_historico_vazio(tmp_path):
    config = Config(arquivo_csv=str(tmp_path / "vazio.csv"), pasta_backup=str(tmp_path / "b"))
    import pandas as pd
    df_vazio = pd.DataFrame(columns=["Data", "Sorteio", "N1", "N2", "N3", "N4", "N5"])
    atraso = atrasos_por_numero(df_vazio)
    assert (atraso == 0).all()


def test_retreinar_com_historico_insuficiente_retorna_none(tmp_path):
    config = Config(
        arquivo_csv=str(tmp_path / "resultados.csv"),
        pasta_backup=str(tmp_path / "backups"),
        minimo_sorteios_para_treino=999,
    )
    analyzer = AnalisadorLotaria(config)
    assert analyzer.retreinar() is None


def test_avaliar_modelo_com_historico_insuficiente(tmp_path):
    config = Config(
        arquivo_csv=str(tmp_path / "resultados.csv"),
        pasta_backup=str(tmp_path / "backups"),
        minimo_sorteios_para_treino=999,
    )
    analyzer = AnalisadorLotaria(config)
    resultado = analyzer.avaliar_modelo()
    assert resultado["mae"] is None


def test_importancia_features_none_antes_de_treinar(config):
    analyzer = AnalisadorLotaria(config)
    assert analyzer.importancia_features() is None


def test_importancia_features_apos_treinar(config):
    analyzer = AnalisadorLotaria(config)
    analyzer.retreinar()
    importancias = analyzer.importancia_features()
    assert importancias is not None
    assert len(importancias) == 6  # numero de features em FEATURE_NOMES


def test_remover_resultado(config):
    repo = RepositorioResultados(config)
    repo.adicionar_resultado("2026-08-10", "Kazola", [11, 22, 33, 44, 55])
    assert repo.remover_resultado("2026-08-10", "Kazola") is True
    assert repo.remover_resultado("2026-08-10", "Kazola") is False


def test_importar_dataframe(config):
    repo = RepositorioResultados(config)
    antes = len(repo.df)
    df = pd.DataFrame([
        {"Data": "2026-08-01", "Sorteio": "Fezada", "N1": 1, "N2": 2, "N3": 3, "N4": 4, "N5": 5},
        {"Data": "2026-08-01", "Sorteio": "Aqueceu", "N1": 6, "N2": 7, "N3": 8, "N4": 9, "N5": 10},
    ])
    n = repo.importar_dataframe(df)
    assert n == 2
    assert len(repo.df) >= antes + 2


def test_pares_frequentes(config):
    analyzer = AnalisadorLotaria(config=config)
    pares = analyzer.pares_frequentes(tamanho=2, top=5)
    assert isinstance(pares, list)
    if pares:
        combo, cnt = pares[0]
        assert len(combo) == 2
        assert cnt >= 1


def test_analisar_bilhete(config):
    analyzer = AnalisadorLotaria(config=config)
    info = analyzer.analisar_bilhete([1, 2, 3, 4, 5])
    assert info["numeros"] == [1, 2, 3, 4, 5]
    assert len(info["detalhe"]) == 5
    assert "vezes_juntos" in info
    assert "overlap_sugestoes_hoje" in info


def test_analisar_bilhete_invalido(config):
    analyzer = AnalisadorLotaria(config=config)
    with pytest.raises(ResultadoInvalidoError):
        analyzer.analisar_bilhete([1, 1, 2, 3, 4])


def test_backtest_sugestoes_devolve_dataframe(config):
    analyzer = AnalisadorLotaria(config=config)
    # Config de teste tem minimo baixo; pode devolver vazio se poucos dados
    df_bt = analyzer.backtest_sugestoes(n_ultimos=5)
    assert isinstance(df_bt, pd.DataFrame)
