"""
lotaria_core.py

Coloquei aqui a lógica de negócio: gravação dos resultados, features e
previsão com Random Forest. As duas interfaces (Tkinter e Streamlit)
importam este módulo para eu não repetir código.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

# Fuso de Angola (WAT, UTC+1) — usei isto para "hoje" e horas reais
FUSO_ANGOLA = ZoneInfo("Africa/Luanda")


def agora_angola() -> datetime:
    """Devolvi a data/hora actual em Angola."""
    return datetime.now(FUSO_ANGOLA)


def hoje_angola() -> str:
    """Devolvi a data de hoje em Angola (YYYY-MM-DD)."""
    return agora_angola().strftime("%Y-%m-%d")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lotaria")

# Centralizei as constantes de domínio para evitar strings mágicas espalhadas
SORTEIOS = ["Fezada", "Aqueceu", "Kazola", "Eskebra"]
COLUNAS_NUMEROS = ["N1", "N2", "N3", "N4", "N5"]
NUMERO_MINIMO = 1
NUMERO_MAXIMO = 90
QTD_NUMEROS_POR_SORTEIO = 5
# Guardei a hora oficial de cada sorteio em Angola (WAT)
HORA_SORTEIO = {
    "Fezada": (10, 0),
    "Aqueceu": (13, 0),
    "Kazola": (16, 0),
    "Eskebra": (19, 0),
}


@dataclass
class Config:
    """Guardei aqui os caminhos e parâmetros do sistema.

    Usei uma dataclass (em vez de constantes de módulo) para eu poder
    injectar caminhos diferentes nos testes, sem tocar no disco real.
    """
    arquivo_csv: str = "resultados_lotaria_angola.csv"
    pasta_backup: str = "backups_lotaria"
    arquivo_sugestoes: str = "sugestoes_bloqueadas.json"
    max_backups: int = 30
    janela_recente: int = 15
    minimo_sorteios_para_treino: int = 8
    n_estimators: int = 120
    max_depth: int = 8
    n_opcoes: int = 4  # quantas sugestões de 5 números por sorteio eu gero


def _dados_exemplo() -> list[tuple]:
    """Devolvi um pequeno conjunto só para a primeira execução,
    quando ainda não existe histórico guardado em disco."""
    return [
        ("2026-08-08", "Fezada", 89, 40, 46, 57, 82),
        ("2026-08-08", "Aqueceu", 86, 85, 38, 41, 24),
        ("2026-08-08", "Kazola", 73, 50, 47, 2, 29),
        ("2026-08-08", "Eskebra", 47, 53, 68, 64, 37),
        ("2026-08-07", "Fezada", 21, 6, 78, 34, 79),
        ("2026-08-07", "Aqueceu", 60, 89, 26, 29, 45),
        ("2026-08-07", "Kazola", 59, 84, 7, 90, 66),
        ("2026-08-07", "Eskebra", 85, 8, 45, 30, 68),
        ("2026-08-06", "Fezada", 73, 64, 43, 86, 40),
        ("2026-08-06", "Aqueceu", 35, 68, 90, 47, 55),
        ("2026-08-06", "Kazola", 29, 40, 28, 82, 32),
        ("2026-08-06", "Eskebra", 74, 40, 47, 25, 48),
    ]


class ResultadoInvalidoError(ValueError):
    """Levanto este erro quando um resultado falha a validação,
    para o distinguir de outros ValueError no resto do código."""


class RepositorioResultados:
    """Só carrego, valido e persisto os
    resultados em CSV, com backup automatico. Nao sabe nada sobre ML."""

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.pasta_backup, exist_ok=True)
        self.df: pd.DataFrame = self._carregar_ou_criar()

    def _carregar_ou_criar(self) -> pd.DataFrame:
        caminho = Path(self.config.arquivo_csv)
        if caminho.exists():
            try:
                df = pd.read_csv(caminho)
                df["Data"] = pd.to_datetime(df["Data"])
                self._validar_colunas(df)
                return df.sort_values(["Data", "Sorteio"]).reset_index(drop=True)
            except Exception:
                logger.exception(
                    "Falha ao ler %s; iniciando com dados de exemplo.", caminho
                )
        return self._criar_dados_iniciais()

    @staticmethod
    def _validar_colunas(df: pd.DataFrame) -> None:
        esperadas = {"Data", "Sorteio", *COLUNAS_NUMEROS}
        faltantes = esperadas - set(df.columns)
        if faltantes:
            raise ValueError(f"Colunas ausentes no CSV: {faltantes}")

    def _criar_dados_iniciais(self) -> pd.DataFrame:
        df = pd.DataFrame(_dados_exemplo(), columns=["Data", "Sorteio", *COLUNAS_NUMEROS])
        df["Data"] = pd.to_datetime(df["Data"])
        self._salvar(df)
        return df

    def _salvar(self, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self.config.arquivo_csv, index=False)
        except OSError:
            logger.exception("Falha ao salvar %s", self.config.arquivo_csv)
            raise

    def fazer_backup(self) -> None:
        """Copiei o CSV actual para a pasta de backups e removi os
        backups mais antigos além do limite configurado. Falhas de
        backup sao logadas mas nao interrompem o fluxo principal."""
        caminho = Path(self.config.arquivo_csv)
        if not caminho.exists():
            return
        try:
            nome = f"{self.config.pasta_backup}/backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
            shutil.copy(caminho, nome)
            self._limpar_backups_antigos()
        except OSError:
            logger.exception("Falha ao criar backup de %s", caminho)

    def _limpar_backups_antigos(self) -> None:
        backups = sorted(
            f for f in os.listdir(self.config.pasta_backup) if f.endswith(".csv")
        )
        excedentes = backups[: max(0, len(backups) - self.config.max_backups)]
        for antigo in excedentes:
            os.remove(os.path.join(self.config.pasta_backup, antigo))

    @staticmethod
    def validar_resultado(sorteio: str, numeros: list[int]) -> None:
        """Centralizei a validação: uso-a tanto na camada de dados
        como nas UIs, para a regra não ficar só na interface."""
        if sorteio not in SORTEIOS:
            raise ResultadoInvalidoError(f"Sorteio invalido: {sorteio!r}")
        if len(numeros) != QTD_NUMEROS_POR_SORTEIO:
            raise ResultadoInvalidoError(
                f"Sao esperados {QTD_NUMEROS_POR_SORTEIO} numeros."
            )
        if len(set(numeros)) != QTD_NUMEROS_POR_SORTEIO:
            raise ResultadoInvalidoError("Os numeros devem ser unicos.")
        if any(not (NUMERO_MINIMO <= n <= NUMERO_MAXIMO) for n in numeros):
            raise ResultadoInvalidoError(
                f"Numeros devem estar entre {NUMERO_MINIMO} e {NUMERO_MAXIMO}."
            )

    def adicionar_resultado(self, data: str, sorteio: str, numeros: list[int]) -> None:
        """Validei e adicionei um novo resultado. Se já existir um
        para a mesma data/sorteio, substituo-o."""
        self.validar_resultado(sorteio, numeros)
        self.fazer_backup()
        novo = pd.DataFrame([{
            "Data": pd.to_datetime(data),
            "Sorteio": sorteio,
            **{col: int(n) for col, n in zip(COLUNAS_NUMEROS, numeros)},
        }])
        self.df = (
            pd.concat([self.df, novo], ignore_index=True)
            .drop_duplicates(subset=["Data", "Sorteio"], keep="last")
            .sort_values(["Data", "Sorteio"])
            .reset_index(drop=True)
        )
        self._salvar(self.df)

    def remover_resultado(self, data: str, sorteio: str) -> bool:
        """Removi o resultado da data/sorteio indicados. Devolvi True se removi."""
        if sorteio not in SORTEIOS:
            raise ResultadoInvalidoError(f"Sorteio invalido: {sorteio!r}")
        self.fazer_backup()
        antes = len(self.df)
        data_ts = pd.to_datetime(data)
        self.df = self.df[
            ~((self.df["Data"] == data_ts) & (self.df["Sorteio"] == sorteio))
        ].reset_index(drop=True)
        if len(self.df) < antes:
            self._salvar(self.df)
            return True
        return False

    def importar_dataframe(self, df_novo: pd.DataFrame) -> int:
        """Importei várias linhas (colunas Data, Sorteio, N1..N5). Devolvi quantas gravei."""
        obrigatorias = {"Data", "Sorteio", *COLUNAS_NUMEROS}
        if not obrigatorias.issubset(set(df_novo.columns)):
            raise ResultadoInvalidoError(
                f"CSV precisa das colunas: {sorted(obrigatorias)}"
            )
        gravados = 0
        for _, row in df_novo.iterrows():
            try:
                data = pd.to_datetime(row["Data"]).strftime("%Y-%m-%d")
                nums = [int(row[c]) for c in COLUNAS_NUMEROS]
                self.adicionar_resultado(data, str(row["Sorteio"]), nums)
                gravados += 1
            except (ResultadoInvalidoError, ValueError, TypeError) as exc:
                logger.warning("Linha ignorada no import: %s (%s)", row.to_dict(), exc)
        return gravados


def frequencias_por_numero(df: pd.DataFrame) -> pd.Series:
    """Calculei a frequência de cada número (1-90) no histórico dado,
    de uma vez para todos os numeros (mais rapido que chamar
    calcular_features 90 vezes quando so a frequencia interessa,
    como na Grelha 1-90 ou nos graficos de Analise por Sorteio)."""
    if df.empty:
        return pd.Series(0, index=range(NUMERO_MINIMO, NUMERO_MAXIMO + 1))
    matriz = df[COLUNAS_NUMEROS].to_numpy().ravel()
    contagem = np.bincount(matriz, minlength=NUMERO_MAXIMO + 1)
    return pd.Series(
        contagem[NUMERO_MINIMO:NUMERO_MAXIMO + 1],
        index=range(NUMERO_MINIMO, NUMERO_MAXIMO + 1),
    )


def atrasos_por_numero(df: pd.DataFrame) -> pd.Series:
    """Calculei o atraso (sorteios desde a última aparição) de cada número (1-90)."""
    total = len(df)
    if total == 0:
        return pd.Series(0, index=range(NUMERO_MINIMO, NUMERO_MAXIMO + 1))
    matriz = df[COLUNAS_NUMEROS].to_numpy()
    atrasos = {}
    for num in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
        idx = np.flatnonzero((matriz == num).any(axis=1))
        atrasos[num] = total if idx.size == 0 else total - 1 - idx[-1]
    return pd.Series(atrasos)


FEATURE_NOMES = [
    "Frequencia relativa",
    "Frequencia recente (janela)",
    "Atraso normalizado",
    "Peso temporal",
    "Posicao media",
    "Par/Impar",
]


def calcular_features(df: pd.DataFrame, numero: int, janela_recente: int = 15) -> np.ndarray:
    """Calculei o vector de features estatísticas de um número num
    histórico de sorteios.

    Implementei em numpy de forma vectorizada: a versão antiga usava
    iterrows() e listas Python, o que era O(n) em Python puro por número.
    Aqui comparo as 5 colunas de uma vez — bem mais rápido no treino
    (avalio os 90 números em cada linha do histórico).
    """
    if df.empty:
        return np.zeros(6)

    matriz = df[COLUNAS_NUMEROS].to_numpy()
    total = len(df)

    ocorrencias = matriz == numero  # shape (total, 5), True onde o numero aparece
    linhas_com_numero = ocorrencias.any(axis=1)

    freq_total = linhas_com_numero.sum()

    recentes_mask = linhas_com_numero[-janela_recente:]
    freq_recente = recentes_mask.sum()

    indices_com_numero = np.flatnonzero(linhas_com_numero)
    atraso = total if indices_com_numero.size == 0 else total - 1 - indices_com_numero[-1]

    pesos = (np.arange(total) + 1) / total
    peso_recente = pesos[linhas_com_numero].sum()

    if indices_com_numero.size:
        posicoes = np.argmax(ocorrencias[indices_com_numero], axis=1) + 1
        media_posicao = posicoes.mean()
    else:
        media_posicao = 3.0

    return np.array([
        freq_total / max(total, 1),
        freq_recente / janela_recente,
        atraso / max(total, 1),
        peso_recente,
        media_posicao / QTD_NUMEROS_POR_SORTEIO,
        1.0 if numero % 2 == 0 else 0.0,
    ])


class PreditorLotaria:
    """Treino modelos Random Forest por sorteio e gero rankings de
    números, combinando o modelo com estatísticas simples."""

    def __init__(self, config: Config):
        self.config = config
        self._modelos: dict[str, RandomForestRegressor] = {}

    def limpar_cache(self) -> None:
        """Chamo isto sempre que o histórico de dados muda,
        para forçar o retreino na próxima previsão."""
        self._modelos = {}

    def _subconjunto(self, df: pd.DataFrame, sorteio: Optional[str]) -> tuple[pd.DataFrame, str]:
        if sorteio:
            return df[df["Sorteio"] == sorteio].copy(), sorteio
        return df.copy(), "Geral"

    def _construir_dataset(self, df_local: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Monto X/y de treino: para cada sorteio histórico (a partir do
        5.º), calculo as features de cada um dos 90 números usando so o
        historico anterior aquele sorteio, com alvo 1 se o numero saiu
        naquele sorteio e 0 caso contrario. Extraido do treino para
        poder ser reutilizado tambem na avaliacao (MAE)."""
        X, y = [], []
        for i in range(5, len(df_local)):
            historico = df_local.iloc[:i]
            atual = df_local.iloc[i]
            sorteados = set(atual[COLUNAS_NUMEROS])
            for num in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
                X.append(calcular_features(historico, num, self.config.janela_recente))
                y.append(1.0 if num in sorteados else 0.0)
        return np.array(X), np.array(y)

    def treinar(self, df: pd.DataFrame, sorteio: Optional[str] = None) -> Optional[RandomForestRegressor]:
        df_local, chave = self._subconjunto(df, sorteio)

        if len(df_local) < self.config.minimo_sorteios_para_treino:
            logger.info(
                "Historico insuficiente para treinar '%s' (%d sorteios); "
                "minimo exigido: %d.",
                chave, len(df_local), self.config.minimo_sorteios_para_treino,
            )
            return None

        X, y = self._construir_dataset(df_local)

        modelo = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X, y)
        self._modelos[chave] = modelo
        return modelo

    def importancias(self, sorteio: Optional[str] = None) -> Optional[pd.Series]:
        """Devolvo a importância de cada feature no último modelo
        que treinei para esse sorteio (ou None se ainda não houver
        modelo treinado)."""
        chave = sorteio or "Geral"
        modelo = self._modelos.get(chave)
        if modelo is None or not hasattr(modelo, "feature_importances_"):
            return None
        return pd.Series(modelo.feature_importances_, index=FEATURE_NOMES).sort_values()

    def avaliar(self, df: pd.DataFrame, sorteio: Optional[str] = None) -> dict:
        """Avalio o erro do modelo (MAE) com um split temporal: treino
        nos primeiros 80% dos exemplos e meço o erro nos últimos 20%,
        sem embaralhar (shuffle=False) para nao vazar informacao do
        futuro para o passado, o que inflaria artificialmente a
        precisao aparente do modelo."""
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        df_local, _ = self._subconjunto(df, sorteio)
        if len(df_local) < self.config.minimo_sorteios_para_treino:
            return {"mae": None, "n_amostras": 0}

        X, y = self._construir_dataset(df_local)
        if len(X) < 50:
            return {"mae": None, "n_amostras": len(X)}

        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        modelo = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X_treino, y_treino)
        mae = mean_absolute_error(y_teste, modelo.predict(X_teste))
        return {"mae": float(mae), "n_amostras": len(X)}

    def prever(
        self,
        df: pd.DataFrame,
        sorteio: Optional[str] = None,
        qtd: int = 5,
        df_extra: Optional[pd.DataFrame] = None,
    ) -> tuple[list[int], dict[int, float]]:
        """Devolvo (números_previstos_ordenados, scores_de_todos_os_números).

        df_extra: resultados de sorteios anteriores do mesmo dia (para
        dependencia sequencial). Sao anexados ao historico local apenas
        para calculo de features e frequencia; o modelo continua sendo
        o treinado no historico normal do sorteio.
        """
        df_local, chave = self._subconjunto(df, sorteio)

        if df_extra is not None and not df_extra.empty:
            # Anexa os resultados do dia (sorteios anteriores) para que
            # frequencia/atraso reflitam o que ja saiu hoje.
            df_local = pd.concat([df_local, df_extra], ignore_index=True)
            df_local = df_local.sort_values("Data").reset_index(drop=True)

        modelo = self._modelos.get(chave)
        if modelo is None:
            modelo = self.treinar(df, sorteio)

        total = max(len(df_local), 1)
        matriz = df_local[COLUNAS_NUMEROS].to_numpy() if not df_local.empty else np.empty((0, 5))

        scores: dict[int, float] = {}
        for num in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
            features = calcular_features(df_local, num, self.config.janela_recente)

            score_ml = float(modelo.predict([features])[0]) if modelo is not None else 0.0

            freq = (matriz == num).any(axis=1).sum() / total if matriz.size else 0.0
            score_atraso = features[2]  # ja normalizado dentro de calcular_features

            scores[num] = (score_ml * 0.65) + (freq * 0.25) + (score_atraso * 0.10)

        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        previsao = sorted(num for num, _ in ranking[:qtd])
        return previsao, scores

    def prever_n_opcoes(
        self,
        df: pd.DataFrame,
        sorteio: Optional[str] = None,
        df_extra: Optional[pd.DataFrame] = None,
        n_opcoes: int = 4,
    ) -> list[list[int]]:
        """Gero N sugestões de 5 números:
        - Opção 1: top 5 do ranking
        - Opção 2: 6.º ao 10.º
        - ...
        Devolvo cada grupo ordenado numericamente.
        """
        qtd = max(1, int(n_opcoes)) * QTD_NUMEROS_POR_SORTEIO
        _, scores = self.prever(df, sorteio=sorteio, qtd=qtd, df_extra=df_extra)
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top = [num for num, _ in ranking[:qtd]]

        opcoes = []
        for i in range(0, qtd, QTD_NUMEROS_POR_SORTEIO):
            grupo = sorted(top[i : i + QTD_NUMEROS_POR_SORTEIO])
            opcoes.append(grupo)
        return opcoes

    def prever_tres_opcoes(
        self,
        df: pd.DataFrame,
        sorteio: Optional[str] = None,
        df_extra: Optional[pd.DataFrame] = None,
    ) -> list[list[int]]:
        """Mantive por compatibilidade: delego para prever_n_opcoes com 4 opções."""
        return self.prever_n_opcoes(df, sorteio=sorteio, df_extra=df_extra, n_opcoes=4)


@runtime_checkable
class RepositorioBase(Protocol):
    """Contrato que qualquer backend de persistencia precisa cumprir.

    Definir isso como Protocol (em vez de herdar de uma classe base)
    permite trocar CSV <-> Supabase <-> outro banco no futuro sem
    alterar AnalisadorLotaria nem as UIs: qualquer objeto com esses
    dois membros serve.
    """

    @property
    def df(self) -> pd.DataFrame: ...

    def adicionar_resultado(self, data: str, sorteio: str, numeros: list[int]) -> None: ...

    def remover_resultado(self, data: str, sorteio: str) -> bool: ...

    def importar_dataframe(self, df_novo: pd.DataFrame) -> int: ...


class RepositorioSupabase:
    """Persisti os dados no Supabase (Postgres gerido) em vez de CSV.

    Assim evito o disco efémero em plataformas como Streamlit Community
    Cloud, onde qualquer ficheiro escrito se perde a cada reinício.

    Pré-requisito: as tabelas têm de existir no projecto Supabase.
    Ver supabase/schema.sql para o script de criação.
    """

    TABELA = "resultados"

    def __init__(self, url: str, key: str):
        # Import local: assim o pacote 'supabase' só precisa estar
        # instalado quando este backend é realmente usado, sem virar
        # dependência obrigatória para quem só usa o backend CSV.
        from supabase import create_client

        self._client = create_client(url, key)

    @property
    def df(self) -> pd.DataFrame:
        resposta = (
            self._client.table(self.TABELA)
            .select("data, sorteio, n1, n2, n3, n4, n5")
            .order("data")
            .order("sorteio")
            .execute()
        )
        linhas = resposta.data
        if not linhas:
            return pd.DataFrame(columns=["Data", "Sorteio", *COLUNAS_NUMEROS])

        df = pd.DataFrame(linhas).rename(columns={
            "data": "Data", "sorteio": "Sorteio",
            "n1": "N1", "n2": "N2", "n3": "N3", "n4": "N4", "n5": "N5",
        })
        df["Data"] = pd.to_datetime(df["Data"])
        return df.sort_values(["Data", "Sorteio"]).reset_index(drop=True)

    def adicionar_resultado(self, data: str, sorteio: str, numeros: list[int]) -> None:
        RepositorioResultados.validar_resultado(sorteio, numeros)
        registro = {
            "data": data,
            "sorteio": sorteio,
            **{f"n{i + 1}": int(n) for i, n in enumerate(numeros)},
        }
        try:
            self._client.table(self.TABELA).upsert(
                registro, on_conflict="data,sorteio"
            ).execute()
        except Exception:
            logger.exception("Falha ao gravar resultado no Supabase.")
            raise

    def remover_resultado(self, data: str, sorteio: str) -> bool:
        if sorteio not in SORTEIOS:
            raise ResultadoInvalidoError(f"Sorteio invalido: {sorteio!r}")
        try:
            resp = (
                self._client.table(self.TABELA)
                .delete()
                .eq("data", data)
                .eq("sorteio", sorteio)
                .execute()
            )
            return bool(resp.data)
        except Exception:
            logger.exception("Falha ao remover resultado no Supabase.")
            raise

    def importar_dataframe(self, df_novo: pd.DataFrame) -> int:
        obrigatorias = {"Data", "Sorteio", *COLUNAS_NUMEROS}
        if not obrigatorias.issubset(set(df_novo.columns)):
            raise ResultadoInvalidoError(
                f"CSV precisa das colunas: {sorted(obrigatorias)}"
            )
        gravados = 0
        for _, row in df_novo.iterrows():
            try:
                data = pd.to_datetime(row["Data"]).strftime("%Y-%m-%d")
                nums = [int(row[c]) for c in COLUNAS_NUMEROS]
                self.adicionar_resultado(data, str(row["Sorteio"]), nums)
                gravados += 1
            except (ResultadoInvalidoError, ValueError, TypeError) as exc:
                logger.warning("Linha ignorada no import: %s (%s)", dict(row), exc)
        return gravados


def criar_repositorio(
    config: Optional[Config] = None,
    backend: str = "csv",
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
) -> RepositorioBase:
    """Centralizei aqui a escolha de backend, para as UIs usarem.

    backend='csv'       → ficheiro local (bom offline / Termux)
    backend='supabase'  → Postgres remoto (preciso no deploy,
                          porque o disco do servidor é efémero)
    """
    if backend == "supabase":
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL e SUPABASE_KEY sao obrigatorios para o backend 'supabase'."
            )
        return RepositorioSupabase(supabase_url, supabase_key)
    return RepositorioResultados(config or Config())



# ---------------------------------------------------------------------------
# Armazenamento de sugestões bloqueadas (JSON local ou Supabase)
# ---------------------------------------------------------------------------

class ArmazenamentoSugestoesJSON:
    """Guardei as sugestões bloqueadas em JSON local (offline / Termux)."""

    def __init__(self, caminho: str | Path):
        self._caminho = Path(caminho)

    def carregar_tudo(self) -> dict:
        if not self._caminho.exists():
            return {}
        try:
            with open(self._caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
            return dados if isinstance(dados, dict) else {}
        except (json.JSONDecodeError, OSError):
            logger.exception("Falha ao ler %s", self._caminho)
            return {}

    def _salvar_tudo(self, dados: dict) -> None:
        try:
            with open(self._caminho, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
        except OSError:
            logger.exception("Falha ao gravar %s", self._caminho)

    def obter(self, data: str, sorteio: str) -> Optional[dict]:
        dados = self.carregar_tudo()
        return AnalisadorLotaria._normalizar_entrada_sugestoes(
            dados.get(data, {}).get(sorteio)
        )

    def gravar(
        self, data: str, sorteio: str, opcoes: list[list[int]], bloqueado_em: Optional[str]
    ) -> bool:
        """Só gravo se ainda não existir. Devolvi True se gravei."""
        dados = self.carregar_tudo()
        if data not in dados:
            dados[data] = {}
        if sorteio in dados[data]:
            return False
        dados[data][sorteio] = {
            "opcoes": [[int(n) for n in op] for op in opcoes],
            "bloqueado_em": bloqueado_em,
        }
        self._salvar_tudo(dados)
        return True


class ArmazenamentoSugestoesSupabase:
    """Guardei as sugestões bloqueadas na tabela do Postgres (Supabase)."""

    TABELA = "sugestoes_bloqueadas"

    def __init__(self, client):
        self._client = client

    def carregar_tudo(self) -> dict:
        """Devolvi no mesmo formato do JSON: {data: {sorteio: {opcoes, bloqueado_em}}}."""
        try:
            resp = (
                self._client.table(self.TABELA)
                .select("data, sorteio, opcoes, bloqueado_em")
                .order("data", desc=True)
                .execute()
            )
        except Exception:
            logger.exception("Falha ao ler sugestões no Supabase.")
            return {}

        dados: dict = {}
        for row in resp.data or []:
            d = row["data"]
            if hasattr(d, "isoformat"):
                d = d.isoformat()[:10]
            d = str(d)[:10]
            s = row["sorteio"]
            if d not in dados:
                dados[d] = {}
            dados[d][s] = {
                "opcoes": row["opcoes"],
                "bloqueado_em": row.get("bloqueado_em"),
            }
        return dados

    def obter(self, data: str, sorteio: str) -> Optional[dict]:
        try:
            resp = (
                self._client.table(self.TABELA)
                .select("opcoes, bloqueado_em")
                .eq("data", data)
                .eq("sorteio", sorteio)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception("Falha ao obter sugestão %s/%s no Supabase", data, sorteio)
            return None
        if not resp.data:
            return None
        row = resp.data[0]
        return AnalisadorLotaria._normalizar_entrada_sugestoes({
            "opcoes": row["opcoes"],
            "bloqueado_em": row.get("bloqueado_em"),
        })

    def gravar(
        self, data: str, sorteio: str, opcoes: list[list[int]], bloqueado_em: Optional[str]
    ) -> bool:
        """Só gravo se ainda não existir (imutável). Devolvi True se gravei."""
        if self.obter(data, sorteio) is not None:
            return False
        registro = {
            "data": data,
            "sorteio": sorteio,
            "opcoes": [[int(n) for n in op] for op in opcoes],
            "bloqueado_em": bloqueado_em,
        }
        try:
            # insert simples: se já existir (race), a PK impede; tratamos como ok
            self._client.table(self.TABELA).insert(registro).execute()
            return True
        except Exception as exc:
            # conflito de PK = já existia → não sobrescrever
            msg = str(exc).lower()
            if "duplicate" in msg or "unique" in msg or "23505" in msg:
                return False
            logger.exception("Falha ao gravar sugestão %s/%s no Supabase", data, sorteio)
            raise


def criar_armazenamento_sugestoes(
    config: Optional[Config] = None,
    repositorio: Optional[object] = None,
):
    """Escolhi JSON local ou Supabase conforme o repositório de resultados."""
    config = config or Config()
    if isinstance(repositorio, RepositorioSupabase):
        return ArmazenamentoSugestoesSupabase(repositorio._client)
    return ArmazenamentoSugestoesJSON(config.arquivo_sugestoes)


class AnalisadorLotaria:
    """Juntei o repositório de dados com o preditor numa fachada simples
    que as duas interfaces (Tkinter/Streamlit) consomem, sem precisarem
    dos detalhes internos.

    Posso injectar o repositório (Supabase, fakes de teste, etc.);
    se não receber nenhum, uso CSV local por omissão.
    """

    def __init__(self, config: Optional[Config] = None, repositorio: Optional[RepositorioBase] = None):
        self.config = config or Config()
        self._repositorio = repositorio or RepositorioResultados(self.config)
        self._preditor = PreditorLotaria(self.config)
        self._sugestoes = criar_armazenamento_sugestoes(self.config, self._repositorio)

    @property
    def df(self) -> pd.DataFrame:
        return self._repositorio.df

    def adicionar_resultado(self, data: str, sorteio: str, numeros: list[int]) -> None:
        # Antes de gravar o resultado: se ainda não houver sugestões
        # bloqueadas para este data+sorteio, calcula e congela-as com o
        # histórico disponível até agora (sem este resultado). Assim as
        # sugestões que já tinham sido mostradas não mudam.
        if self.obter_sugestoes_bloqueadas(data, sorteio) is None:
            try:
                opcoes = self._calcular_opcoes_sorteio(data, sorteio)
                self.bloquear_sugestoes(data, sorteio, opcoes)
            except Exception:
                logger.exception(
                    "Não foi possível bloquear sugestões para %s / %s", data, sorteio
                )

        self._repositorio.adicionar_resultado(data, sorteio, numeros)
        self._preditor.limpar_cache()

    def remover_resultado(self, data: str, sorteio: str) -> bool:
        ok = self._repositorio.remover_resultado(data, sorteio)
        if ok:
            self._preditor.limpar_cache()
        return ok

    def importar_dataframe(self, df_novo: pd.DataFrame) -> int:
        n = self._repositorio.importar_dataframe(df_novo)
        if n:
            self._preditor.limpar_cache()
        return n

    def obter_resultado(
        self, data: str, sorteio: str
    ) -> Optional[list[int]]:
        """Devolvi os 5 números já registados para data+sorteio, ou None."""
        if self.df.empty:
            return None
        if sorteio not in SORTEIOS:
            return None
        data_ts = pd.to_datetime(data)
        match = self.df[
            (self.df["Data"] == data_ts) & (self.df["Sorteio"] == sorteio)
        ]
        if match.empty:
            return None
        row = match.iloc[0]
        return [int(row[c]) for c in COLUNAS_NUMEROS]

    def verificar_conflito_registo(
        self, data: str, sorteio: str, numeros: list[int]
    ) -> dict:
        """Verifiquei se gravar estes números entra em conflito com o existente.

        Devolvi:
          status: 'novo' | 'igual' | 'diferente'
          existente: list[int] | None
          novos: list[int]
          mensagem: str
        """
        RepositorioResultados.validar_resultado(sorteio, list(numeros))
        novos = [int(n) for n in numeros]
        existente = self.obter_resultado(data, sorteio)
        if existente is None:
            return {
                "status": "novo",
                "existente": None,
                "novos": novos,
                "mensagem": f"Novo registo: {data} · {sorteio}.",
            }
        if sorted(existente) == sorted(novos):
            return {
                "status": "igual",
                "existente": existente,
                "novos": novos,
                "mensagem": (
                    f"Já existe o mesmo resultado para {data} · {sorteio}: "
                    f"{', '.join(map(str, existente))}."
                ),
            }
        return {
            "status": "diferente",
            "existente": existente,
            "novos": novos,
            "mensagem": (
                f"CONFLITO: {data} · {sorteio} já tem números diferentes.\n"
                f"  Actual: {', '.join(map(str, existente))}\n"
                f"  Novo:   {', '.join(map(str, novos))}\n"
                f"Gravar substitui o registo anterior."
            ),
        }

    def relatorio_qualidade(
        self, data_inicio: Optional[str] = None, data_fim: Optional[str] = None
    ) -> dict:
        """Monte o relatório de qualidade dos dados na base.

        Verifico:
        - linhas com números inválidos / repetidos dentro do sorteio
        - pares data+sorteio duplicados (não deviam existir após upsert)
        - sorteios em falta por dia (dos 4 oficiais)
        - datas incompletas (menos de 4 sorteios)
        """
        vazio = {
            "n_registos": 0,
            "linhas_invalidas": [],
            "duplicados_chave": [],
            "datas_incompletas": [],
            "sorteios_em_falta": [],
            "resumo": {
                "invalidas": 0,
                "duplicados": 0,
                "datas_incompletas": 0,
                "sorteios_em_falta": 0,
            },
        }
        if self.df.empty:
            return vazio

        df = self.df.copy()
        df["_data"] = df["Data"].dt.strftime("%Y-%m-%d")

        if data_inicio:
            df = df[df["_data"] >= data_inicio]
        if data_fim:
            df = df[df["_data"] <= data_fim]

        if df.empty:
            return vazio

        # 1) Linhas inválidas (números fora de 1-90, repetidos, sorteio desconhecido)
        invalidas = []
        for idx, row in df.iterrows():
            problemas = []
            s = row["Sorteio"]
            if s not in SORTEIOS:
                problemas.append(f"sorteio desconhecido: {s!r}")
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            if len(set(nums)) != QTD_NUMEROS_POR_SORTEIO:
                problemas.append(f"números repetidos: {nums}")
            for n in nums:
                if not (NUMERO_MINIMO <= n <= NUMERO_MAXIMO):
                    problemas.append(f"número fora do intervalo: {n}")
            if problemas:
                invalidas.append({
                    "data": row["_data"],
                    "sorteio": s,
                    "numeros": nums,
                    "problemas": problemas,
                })

        # 2) Duplicados de chave data+sorteio (com números eventualmente diferentes)
        contagem = df.groupby(["_data", "Sorteio"]).size()
        dups_chave = contagem[contagem > 1]
        duplicados = []
        for (d, s), n in dups_chave.items():
            sub = df[(df["_data"] == d) & (df["Sorteio"] == s)]
            variantes = [
                [int(r[c]) for c in COLUNAS_NUMEROS] for _, r in sub.iterrows()
            ]
            # únicos conjuntos de números
            unicos = []
            for v in variantes:
                if sorted(v) not in [sorted(u) for u in unicos]:
                    unicos.append(v)
            duplicados.append({
                "data": d,
                "sorteio": s,
                "ocorrencias": int(n),
                "variantes": unicos,
                "numeros_diferentes": len(unicos) > 1,
            })

        # 3) Sorteios em falta e datas incompletas
        datas = sorted(df["_data"].unique())
        sorteios_falta = []
        datas_incompletas = []
        for d in datas:
            presentes = set(df.loc[df["_data"] == d, "Sorteio"].tolist())
            em_falta = [s for s in SORTEIOS if s not in presentes]
            if em_falta:
                datas_incompletas.append({
                    "data": d,
                    "presentes": sorted(presentes),
                    "em_falta": em_falta,
                    "n_presentes": len(presentes),
                })
                for s in em_falta:
                    h, m = HORA_SORTEIO.get(s, (0, 0))
                    sorteios_falta.append({
                        "data": d,
                        "sorteio": s,
                        "hora": f"{h:02d}:{m:02d}",
                    })

        return {
            "n_registos": len(df),
            "linhas_invalidas": invalidas,
            "duplicados_chave": duplicados,
            "datas_incompletas": datas_incompletas,
            "sorteios_em_falta": sorteios_falta,
            "resumo": {
                "invalidas": len(invalidas),
                "duplicados": len(duplicados),
                "datas_incompletas": len(datas_incompletas),
                "sorteios_em_falta": len(sorteios_falta),
            },
        }

    def prever_numeros(
        self, sorteio: Optional[str] = None, qtd: int = 5, df_extra: Optional[pd.DataFrame] = None
    ) -> tuple[list[int], dict[int, float]]:
        return self._preditor.prever(self.df, sorteio, qtd, df_extra=df_extra)

    def prever_todos_sorteios(self) -> dict[str, list[int]]:
        """Conveniência que montei para as duas UIs: resumo com a
        previsão geral + previsão de cada horário.
        Mantive por compatibilidade (1 sugestão de 5 números)."""
        resultado = {"Geral": self.prever_numeros(sorteio=None)[0]}
        for sorteio in SORTEIOS:
            resultado[sorteio] = self.prever_numeros(sorteio=sorteio)[0]
        return resultado

    # ------------------------------------------------------------------
    # Sugestões bloqueadas (congeladas após o resultado ser lançado)
    # ------------------------------------------------------------------

    def _carregar_sugestoes_bloqueadas(self) -> dict:
        """Carreguei no formato unificado {data: {sorteio: {opcoes, bloqueado_em}}}."""
        return self._sugestoes.carregar_tudo()

    @staticmethod
    def _normalizar_entrada_sugestoes(entrada) -> Optional[dict]:
        """Aceitei o formato antigo (lista de listas) ou o novo (dict com opcoes/meta)."""
        if entrada is None:
            return None
        if isinstance(entrada, list):
            if not entrada:
                return None
            return {
                "opcoes": [[int(n) for n in op] for op in entrada],
                "bloqueado_em": None,
            }
        if isinstance(entrada, dict):
            ops = entrada.get("opcoes")
            if not ops:
                return None
            return {
                "opcoes": [[int(n) for n in op] for op in ops],
                "bloqueado_em": entrada.get("bloqueado_em"),
            }
        return None

    def obter_sugestoes_bloqueadas(
        self, data: str, sorteio: str
    ) -> Optional[list[list[int]]]:
        """Devolvi as opções congeladas para data+sorteio, ou None."""
        meta = self.obter_sugestoes_meta(data, sorteio)
        return meta["opcoes"] if meta else None

    def obter_sugestoes_meta(
        self, data: str, sorteio: str
    ) -> Optional[dict]:
        """Devolvi o dict {opcoes, bloqueado_em} ou None."""
        return self._sugestoes.obter(data, sorteio)

    def bloquear_sugestoes(
        self, data: str, sorteio: str, opcoes: list[list[int]]
    ) -> None:
        """Conglei as sugestões de um sorteio (não voltam a ser recalculadas)."""
        bloqueado_em = agora_angola().isoformat(timespec="seconds")
        gravou = self._sugestoes.gravar(data, sorteio, opcoes, bloqueado_em)
        if gravou:
            logger.info("Sugestões bloqueadas para %s / %s", data, sorteio)

    def _df_para_previsao(
        self, data: str, sorteio: str
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Preparei o histórico para prever um sorteio sem 'olhar o futuro'.

        - Excluí o resultado do próprio sorteio nessa data (e dos sorteios
          posteriores do mesmo dia), para as sugestões não mudarem depois
          de o resultado ser lançado.
        - Incluí como df_extra os resultados dos sorteios anteriores do
          mesmo dia (dependência sequencial).
        """
        if self.df.empty:
            return self.df.copy(), None

        df = self.df.copy()
        df["_data_str"] = df["Data"].dt.strftime("%Y-%m-%d")

        idx_sorteio = SORTEIOS.index(sorteio) if sorteio in SORTEIOS else -1
        sorteios_excluir = set(SORTEIOS[idx_sorteio:]) if idx_sorteio >= 0 else {sorteio}

        # Base: tudo excepto este sorteio e os posteriores na mesma data
        mask_excluir = (df["_data_str"] == data) & (df["Sorteio"].isin(sorteios_excluir))
        df_base = df.loc[~mask_excluir].drop(columns=["_data_str"])

        # Extra: sorteios anteriores do mesmo dia já registados
        anteriores = SORTEIOS[:idx_sorteio] if idx_sorteio > 0 else []
        if anteriores:
            mask_extra = (df["_data_str"] == data) & (df["Sorteio"].isin(anteriores))
            df_extra = df.loc[mask_extra].drop(columns=["_data_str"])
            if df_extra.empty:
                df_extra = None
        else:
            df_extra = None

        return df_base, df_extra

    def _calcular_opcoes_sorteio(
        self, data: str, sorteio: str
    ) -> list[list[int]]:
        """Calculei N opções para um sorteio sem usar o resultado desse sorteio."""
        df_base, df_extra = self._df_para_previsao(data, sorteio)
        return self._preditor.prever_n_opcoes(
            df_base,
            sorteio=sorteio,
            df_extra=df_extra,
            n_opcoes=self.config.n_opcoes,
        )

    def prever_opcoes_por_sorteio(
        self, data: Optional[str] = None
    ) -> dict[str, list[list[int]]]:
        """Devolvi as sugestões de cada sorteio para a data (hoje por omissão).

        Regras que segui:
        - Se já existem sugestões bloqueadas para data+sorteio → devolvo-as
          (não recalculo nunca mais).
        - Caso contrário calculo uma vez (sem usar o resultado desse sorteio
          nem dos posteriores do mesmo dia) e **congelo de imediato**.
        - Assim, depois de o resultado ser lançado, as sugestões que o
          utilizador já viu não mudam; os novos dados só melhoram os
          sorteios seguintes.
        """
        data = data or hoje_angola()
        resultado: dict[str, list[list[int]]] = {}

        for sorteio in SORTEIOS:
            bloqueadas = self.obter_sugestoes_bloqueadas(data, sorteio)
            if bloqueadas is not None:
                resultado[sorteio] = bloqueadas
                continue

            opcoes = self._calcular_opcoes_sorteio(data, sorteio)
            # Congelo na primeira vez que as calculo (ficam imutáveis a partir daqui)
            self.bloquear_sugestoes(data, sorteio, opcoes)
            resultado[sorteio] = opcoes

        return resultado

    def prever_tres_opcoes_por_sorteio(self) -> dict[str, list[list[int]]]:
        """Mantive este alias por compatibilidade → prever_opcoes_por_sorteio()."""
        return self.prever_opcoes_por_sorteio()

    def frequencias(self, sorteio: Optional[str] = None) -> pd.Series:
        """Calculei a frequência de cada número (1-90), opcionalmente filtrada
        por sorteio. Uso isto na Grelha 1-90 e na Análise por Sorteio."""
        df_local = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        return frequencias_por_numero(df_local)

    def atrasos(self, sorteio: Optional[str] = None) -> pd.Series:
        """Calculei o atraso de cada número (1-90), opcionalmente filtrado por sorteio."""
        df_local = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        return atrasos_por_numero(df_local)

    def _df_periodo(
        self,
        sorteio: Optional[str] = None,
        meses: Optional[int] = None,
        data_inicio: Optional[str] = None,
        data_fim: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filtrei o histórico por sorteio e/ou janela temporal."""
        df = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        if df.empty:
            return df.copy()
        df = df.copy()
        if meses is not None and meses > 0:
            fim = df["Data"].max()
            inicio = fim - pd.DateOffset(months=int(meses))
            df = df[df["Data"] >= inicio]
        if data_inicio:
            df = df[df["Data"] >= pd.to_datetime(data_inicio)]
        if data_fim:
            df = df[df["Data"] <= pd.to_datetime(data_fim)]
        return df.reset_index(drop=True)

    def frequencias_periodo(
        self,
        sorteio: Optional[str] = None,
        meses: Optional[int] = None,
        data_inicio: Optional[str] = None,
        data_fim: Optional[str] = None,
    ) -> pd.Series:
        """Calculei a frequência absoluta (1-90) num período."""
        return frequencias_por_numero(
            self._df_periodo(sorteio, meses, data_inicio, data_fim)
        )

    def quentes_frios(
        self,
        sorteio: Optional[str] = None,
        meses: Optional[int] = None,
        top: int = 15,
    ) -> dict:
        """Listei os números mais e menos frequentes no período vs o total."""
        freq_total = self.frequencias(sorteio)
        freq_periodo = self.frequencias_periodo(sorteio, meses=meses)
        n_total = int(freq_total.sum() // QTD_NUMEROS_POR_SORTEIO) if freq_total.sum() else 0
        df_p = self._df_periodo(sorteio, meses=meses)
        n_periodo = len(df_p)

        quentes_p = freq_periodo.sort_values(ascending=False).head(top)
        frios_p = freq_periodo.sort_values(ascending=True).head(top)
        # frios: preferir os que saíram menos (incluindo 0)
        quentes_t = freq_total.sort_values(ascending=False).head(top)
        frios_t = freq_total.sort_values(ascending=True).head(top)

        return {
            "n_sorteios_total": n_total if not sorteio else int(
                (self.df["Sorteio"] == sorteio).sum() if not self.df.empty else 0
            ),
            "n_sorteios_periodo": n_periodo,
            "meses": meses,
            "quentes_periodo": [(int(n), int(v)) for n, v in quentes_p.items()],
            "frios_periodo": [(int(n), int(v)) for n, v in frios_p.items()],
            "quentes_total": [(int(n), int(v)) for n, v in quentes_t.items()],
            "frios_total": [(int(n), int(v)) for n, v in frios_t.items()],
            "freq_periodo": freq_periodo,
            "freq_total": freq_total,
        }

    def distribuicao_somas(
        self, sorteio: Optional[str] = None, meses: Optional[int] = None
    ) -> dict:
        """Calculei a soma dos 5 números por sorteio e as estatísticas (aprox. normal)."""
        df = self._df_periodo(sorteio, meses=meses)
        if df.empty:
            return {
                "somas": [],
                "media": None,
                "desvio": None,
                "min": None,
                "max": None,
                "n": 0,
            }
        somas = df[COLUNAS_NUMEROS].sum(axis=1).astype(int)
        return {
            "somas": somas.tolist(),
            "media": float(somas.mean()),
            "desvio": float(somas.std(ddof=0)) if len(somas) > 1 else 0.0,
            "min": int(somas.min()),
            "max": int(somas.max()),
            "n": len(somas),
            "percentis": {
                "p5": float(somas.quantile(0.05)),
                "p25": float(somas.quantile(0.25)),
                "p50": float(somas.quantile(0.50)),
                "p75": float(somas.quantile(0.75)),
                "p95": float(somas.quantile(0.95)),
            },
        }

    def avaliar_soma_combinacao(self, numeros: list[int]) -> dict:
        """Situai a soma de uma combinação face à distribuição histórica."""
        soma = int(sum(numeros))
        dist = self.distribuicao_somas()
        if dist["n"] == 0 or dist["media"] is None:
            return {"soma": soma, "avaliacao": "sem_historico", "mensagem": "Sem histórico."}
        media, desvio = dist["media"], dist["desvio"] or 1.0
        z = (soma - media) / desvio
        p5, p95 = dist["percentis"]["p5"], dist["percentis"]["p95"]
        if soma < p5 or soma > p95:
            faixa = "improvável (fora de ~90% do histórico)"
        elif abs(z) > 1.5:
            faixa = "pouco comum"
        else:
            faixa = "dentro da zona habitual"
        return {
            "soma": soma,
            "media_historica": media,
            "desvio": desvio,
            "z": z,
            "faixa": faixa,
            "percentis": dist["percentis"],
            "mensagem": (
                f"Soma **{soma}** — média histórica {media:.1f} "
                f"(σ={desvio:.1f}). Avaliação: **{faixa}**."
            ),
        }

    def distribuicao_paridade(
        self, sorteio: Optional[str] = None, meses: Optional[int] = None
    ) -> dict:
        """Contei pares/ímpares por sorteio e os padrões (ex.: 3 pares + 2 ímpares)."""
        df = self._df_periodo(sorteio, meses=meses)
        if df.empty:
            return {
                "n": 0,
                "padroes": {},
                "total_pares": 0,
                "total_impares": 0,
                "media_pares_por_sorteio": None,
            }
        padroes: dict[str, int] = {}
        total_pares = 0
        total_impares = 0
        for _, row in df.iterrows():
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            n_pares = sum(1 for n in nums if n % 2 == 0)
            n_imp = 5 - n_pares
            total_pares += n_pares
            total_impares += n_imp
            chave = f"{n_pares} pares + {n_imp} ímpares"
            padroes[chave] = padroes.get(chave, 0) + 1
        # ordenar padrões do mais comum
        padroes_ord = dict(sorted(padroes.items(), key=lambda x: -x[1]))
        return {
            "n": len(df),
            "padroes": padroes_ord,
            "total_pares": total_pares,
            "total_impares": total_impares,
            "media_pares_por_sorteio": total_pares / len(df),
            "pct_pares": 100.0 * total_pares / (total_pares + total_impares),
        }

    def retreinar(self, sorteio: Optional[str] = None) -> Optional[RandomForestRegressor]:
        """Forcei o retreino do modelo (sob demanda), usado pelo botão
        'Treinar agora' na página Modelo ML."""
        return self._preditor.treinar(self.df, sorteio)

    def importancia_features(self, sorteio: Optional[str] = None) -> Optional[pd.Series]:
        return self._preditor.importancias(sorteio)

    def avaliar_modelo(self, sorteio: Optional[str] = None) -> dict:
        return self._preditor.avaliar(self.df, sorteio)

    def pares_frequentes(
        self, sorteio: Optional[str] = None, tamanho: int = 2, top: int = 10
    ) -> list[tuple[tuple[int, ...], int]]:
        """Listei as combinações de `tamanho` números que mais saem juntos no histórico."""
        from collections import Counter
        from itertools import combinations

        df_local = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        if df_local.empty:
            return []
        contador: Counter = Counter()
        for _, row in df_local.iterrows():
            nums = sorted(int(row[c]) for c in COLUNAS_NUMEROS)
            for combo in combinations(nums, tamanho):
                contador[combo] += 1
        return contador.most_common(top)

    def analisar_bilhete(
        self, numeros: list[int], sorteio: Optional[str] = None
    ) -> dict:
        """Analisei 5 números do utilizador face ao histórico e às sugestões de hoje."""
        if sorteio is not None and sorteio not in SORTEIOS:
            raise ResultadoInvalidoError(f"Sorteio invalido: {sorteio!r}")
        if len(numeros) != QTD_NUMEROS_POR_SORTEIO or len(set(numeros)) != QTD_NUMEROS_POR_SORTEIO:
            raise ResultadoInvalidoError("Sao esperados 5 numeros unicos.")
        if any(not (NUMERO_MINIMO <= n <= NUMERO_MAXIMO) for n in numeros):
            raise ResultadoInvalidoError(
                f"Numeros devem estar entre {NUMERO_MINIMO} e {NUMERO_MAXIMO}."
            )

        freq = self.frequencias(sorteio)
        atr = self.atrasos(sorteio)
        df_local = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        conjunto = set(numeros)

        juntos = 0
        ultima_data = None
        for _, row in df_local.iterrows():
            real = {int(row[c]) for c in COLUNAS_NUMEROS}
            if conjunto.issubset(real):
                juntos += 1
                d = row["Data"]
                ultima_data = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

        # quantas vezes cada número saiu
        detalhe = [
            {
                "numero": n,
                "frequencia": int(freq.get(n, 0)),
                "atraso": int(atr.get(n, 0)),
            }
            for n in sorted(numeros)
        ]

        overlap_hoje: dict[str, list[int]] = {}
        if not self.df.empty:
            tres = self.prever_tres_opcoes_por_sorteio()
            alvos = [sorteio] if sorteio else SORTEIOS
            for s in alvos:
                for i, op in enumerate(tres[s], 1):
                    comuns = sorted(conjunto & set(op))
                    if comuns:
                        overlap_hoje[f"{s} · Opção {i}"] = comuns

        return {
            "numeros": sorted(numeros),
            "detalhe": detalhe,
            "vezes_juntos": juntos,
            "ultima_vez_juntos": ultima_data,
            "overlap_sugestoes_hoje": overlap_hoje,
        }

    def backtest_sugestoes(
        self, n_ultimos: int = 15, sorteio: Optional[str] = None
    ) -> pd.DataFrame:
        """Comparei, em cada sorteio passado, a 1.ª sugestão (treino só com
        dados anteriores) com o resultado real. Mostro quantos acertos."""
        df = self.df.sort_values(["Data", "Sorteio"]).reset_index(drop=True)
        if sorteio:
            indices = df.index[df["Sorteio"] == sorteio].tolist()
        else:
            indices = df.index.tolist()
        if not indices:
            return pd.DataFrame()

        alvo = indices[-n_ultimos:]
        linhas = []
        minimo = self.config.minimo_sorteios_para_treino

        for idx in alvo:
            hist = df.iloc[:idx]
            if len(hist) < minimo:
                continue
            row = df.iloc[idx]
            s = row["Sorteio"]
            real = [int(row[c]) for c in COLUNAS_NUMEROS]
            try:
                pred, _ = self._preditor.prever(hist, sorteio=s, qtd=5)
            except Exception:
                continue
            acertos = len(set(pred) & set(real))
            data = row["Data"]
            data_s = data.strftime("%Y-%m-%d") if hasattr(data, "strftime") else str(data)
            linhas.append({
                "Data": data_s,
                "Sorteio": s,
                "Real": ", ".join(map(str, sorted(real))),
                "Sugestão": ", ".join(map(str, sorted(pred))),
                "Acertos": acertos,
            })

        return pd.DataFrame(linhas)

    def resultado_vs_sugestoes_hoje(self) -> dict[str, dict]:
        """Para cada sorteio de hoje já registado, comparei com as opções
        (bloqueadas se existirem — não recalculo depois do lançamento)."""
        return self.comparar_resultados_com_sugestoes(data=hoje_angola())

    def comparar_resultados_com_sugestoes(
        self, data: Optional[str] = None
    ) -> dict[str, dict]:
        """Comparei resultados reais de uma data com as sugestões bloqueadas.

        Devolvi por sorteio:
          real, opcoes, acertos (lista), melhor, numeros_acertados (por opção).
        """
        if self.df.empty:
            return {}
        data = data or hoje_angola()
        df_dia = self.df[self.df["Data"].dt.strftime("%Y-%m-%d") == data]
        if df_dia.empty:
            return {}

        out: dict[str, dict] = {}
        for _, row in df_dia.iterrows():
            s = row["Sorteio"]
            real = sorted(int(row[c]) for c in COLUNAS_NUMEROS)
            real_set = set(real)
            ops = self.obter_sugestoes_bloqueadas(data, s)
            if ops is None:
                # fallback: calcula (e congela) se ainda não houver
                ops = self.prever_opcoes_por_sorteio(data=data).get(s, [])
            acertos_ops = [len(real_set & set(op)) for op in ops]
            nums_acertados = [sorted(real_set & set(op)) for op in ops]
            out[s] = {
                "real": real,
                "opcoes": ops,
                "acertos": acertos_ops,
                "numeros_acertados": nums_acertados,
                "melhor": max(acertos_ops) if acertos_ops else 0,
                "melhor_opcao": (acertos_ops.index(max(acertos_ops)) + 1) if acertos_ops else None,
            }
        return out

    def historico_sugestoes(
        self, limite_dias: Optional[int] = None
    ) -> list[dict]:
        """Monte o histórico completo de sugestões, do mais recente para o mais antigo.

        Cada item traz:
          data, sorteio, hora_sorteio, bloqueado_em, opcoes,
          real (ou None), acertos, numeros_acertados, melhor.
        """
        dados = self._carregar_sugestoes_bloqueadas()
        if not dados:
            return []

        # mapa data+sorteio -> números reais
        reais: dict[tuple[str, str], list[int]] = {}
        if not self.df.empty:
            for _, row in self.df.iterrows():
                d = row["Data"]
                d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                s = row["Sorteio"]
                reais[(d_str, s)] = sorted(int(row[c]) for c in COLUNAS_NUMEROS)

        itens: list[dict] = []
        for data in sorted(dados.keys(), reverse=True):
            bloco = dados[data]
            if not isinstance(bloco, dict):
                continue
            for sorteio in SORTEIOS:
                if sorteio not in bloco:
                    continue
                meta = self._normalizar_entrada_sugestoes(bloco[sorteio])
                if not meta:
                    continue
                ops = meta["opcoes"]
                real = reais.get((data, sorteio))
                acertos = None
                nums_ac = None
                melhor = None
                melhor_op = None
                if real is not None:
                    rset = set(real)
                    acertos = [len(rset & set(op)) for op in ops]
                    nums_ac = [sorted(rset & set(op)) for op in ops]
                    melhor = max(acertos) if acertos else 0
                    melhor_op = acertos.index(melhor) + 1 if acertos else None

                h, m = HORA_SORTEIO.get(sorteio, (0, 0))
                itens.append({
                    "data": data,
                    "sorteio": sorteio,
                    "hora_sorteio": f"{h:02d}:{m:02d}",
                    "bloqueado_em": meta.get("bloqueado_em"),
                    "opcoes": ops,
                    "real": real,
                    "acertos": acertos,
                    "numeros_acertados": nums_ac,
                    "melhor": melhor,
                    "melhor_opcao": melhor_op,
                })

        if limite_dias is not None and limite_dias > 0:
            datas_unicas = []
            for it in itens:
                if it["data"] not in datas_unicas:
                    datas_unicas.append(it["data"])
            datas_ok = set(datas_unicas[:limite_dias])
            itens = [it for it in itens if it["data"] in datas_ok]

        return itens

    def estatisticas_acertos(self) -> dict:
        """Calculei a média de acertos (melhor opção e média das 4) no histórico
        em que já existe resultado real confrontado com as sugestões bloqueadas."""
        hist = self.historico_sugestoes()
        com_resultado = [h for h in hist if h["real"] is not None and h["acertos"]]
        if not com_resultado:
            return {
                "n_sorteios": 0,
                "media_melhor": None,
                "media_todas_opcoes": None,
                "por_sorteio": {},
                "por_dia": {},
            }

        melhores = [h["melhor"] for h in com_resultado]
        todas = [a for h in com_resultado for a in h["acertos"]]

        por_sorteio: dict[str, dict] = {}
        for s in SORTEIOS:
            sub = [h for h in com_resultado if h["sorteio"] == s]
            if not sub:
                continue
            por_sorteio[s] = {
                "n": len(sub),
                "media_melhor": sum(h["melhor"] for h in sub) / len(sub),
            }

        por_dia: dict[str, dict] = {}
        for h in com_resultado:
            d = h["data"]
            if d not in por_dia:
                por_dia[d] = {"melhores": [], "n": 0}
            por_dia[d]["melhores"].append(h["melhor"])
            por_dia[d]["n"] += 1
        for d, info in por_dia.items():
            info["media_melhor"] = sum(info["melhores"]) / len(info["melhores"])
            del info["melhores"]


        return {
            "n_sorteios": len(com_resultado),
            "media_melhor": sum(melhores) / len(melhores),
            "media_todas_opcoes": sum(todas) / len(todas) if todas else None,
            "por_sorteio": por_sorteio,
            "por_dia": dict(sorted(por_dia.items(), reverse=True)),
        }

    # ------------------------------------------------------------------
    # Algoritmos alternativos de geração de sugestões (Fase 7)
    # ------------------------------------------------------------------

    def _padroes_estruturais_referencia(
        self, df: pd.DataFrame
    ) -> dict:
        """Obtive as faixas históricas de soma, paridade e consecutivos para filtrar."""
        if df.empty:
            return {
                "soma_p10": 100,
                "soma_p90": 350,
                "pares_min": 1,
                "pares_max": 4,
                "max_consecutivos": 2,
            }
        somas = df[COLUNAS_NUMEROS].sum(axis=1)
        n_pares_lista = []
        for _, row in df.iterrows():
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            n_pares_lista.append(sum(1 for n in nums if n % 2 == 0))
        return {
            "soma_p10": float(somas.quantile(0.10)),
            "soma_p90": float(somas.quantile(0.90)),
            "pares_min": max(0, int(np.percentile(n_pares_lista, 10))),
            "pares_max": min(5, int(np.percentile(n_pares_lista, 90))),
            "max_consecutivos": 2,
        }

    @staticmethod
    def _contar_consecutivos(numeros: list[int]) -> int:
        s = sorted(numeros)
        melhor = actual = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                actual += 1
                melhor = max(melhor, actual)
            else:
                actual = 1
        return melhor

    def _respeita_padroes(self, numeros: list[int], ref: dict) -> bool:
        soma = sum(numeros)
        if soma < ref["soma_p10"] or soma > ref["soma_p90"]:
            return False
        n_pares = sum(1 for n in numeros if n % 2 == 0)
        if n_pares < ref["pares_min"] or n_pares > ref["pares_max"]:
            return False
        if self._contar_consecutivos(numeros) > ref["max_consecutivos"]:
            return False
        return True

    def gerar_amostra_ponderada(
        self,
        sorteio: Optional[str] = None,
        n_opcoes: int = 4,
        modo: str = "frequencia",
        seed: Optional[int] = None,
    ) -> list[list[int]]:
        """Gero uma amostragem ponderada.

        modo='frequencia'  → peso ∝ frequência (favorece quentes)
        modo='contrarian'  → peso ∝ atraso (favorece frios; crença popular)
        modo='uniforme'    → baseline aleatório
        """
        rng = np.random.default_rng(seed)
        df = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        freq = frequencias_por_numero(df)
        atr = atrasos_por_numero(df)
        numeros = np.arange(NUMERO_MINIMO, NUMERO_MAXIMO + 1)

        if modo == "contrarian":
            pesos = atr.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5
        elif modo == "uniforme":
            pesos = np.ones(len(numeros), dtype=float)
        else:
            pesos = freq.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5

        pesos = pesos / pesos.sum()
        opcoes = []
        tentativas = 0
        while len(opcoes) < n_opcoes and tentativas < n_opcoes * 30:
            tentativas += 1
            escolha = rng.choice(numeros, size=QTD_NUMEROS_POR_SORTEIO, replace=False, p=pesos)
            combo = sorted(int(x) for x in escolha)
            if combo not in opcoes:
                opcoes.append(combo)
        return opcoes

    def gerar_filtragem_estrutural(
        self,
        sorteio: Optional[str] = None,
        n_opcoes: int = 4,
        modo_peso: str = "frequencia",
        seed: Optional[int] = None,
        max_tentativas: int = 2000,
    ) -> list[list[int]]:
        """Gero combinações por amostragem e só aceito as que respeitam
        padrões históricos (soma, paridade, consecutivos)."""
        rng = np.random.default_rng(seed)
        df = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        ref = self._padroes_estruturais_referencia(df)
        freq = frequencias_por_numero(df)
        atr = atrasos_por_numero(df)
        numeros = np.arange(NUMERO_MINIMO, NUMERO_MAXIMO + 1)

        if modo_peso == "contrarian":
            pesos = atr.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5
        elif modo_peso == "uniforme":
            pesos = np.ones(len(numeros), dtype=float)
        else:
            pesos = freq.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5
        pesos = pesos / pesos.sum()

        opcoes: list[list[int]] = []
        for _ in range(max_tentativas):
            if len(opcoes) >= n_opcoes:
                break
            escolha = rng.choice(numeros, size=QTD_NUMEROS_POR_SORTEIO, replace=False, p=pesos)
            combo = sorted(int(x) for x in escolha)
            if combo in opcoes:
                continue
            if self._respeita_padroes(combo, ref):
                opcoes.append(combo)
        return opcoes

    def gerar_ensemble(
        self,
        sorteio: Optional[str] = None,
        n_opcoes: int = 4,
        seed: Optional[int] = None,
    ) -> dict:
        """Combinei frequência, atraso, ML (se disponível) e padrões.

        Devolvi scores compostos e N opções (top fatias do ranking).
        """
        df = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        freq = frequencias_por_numero(df)
        atr = atrasos_por_numero(df)
        total = max(len(df), 1)

        # Scores ML do preditor actual (se houver dados)
        scores_ml = {n: 0.0 for n in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1)}
        if len(self.df) >= self.config.minimo_sorteios_para_treino:
            try:
                _, sml = self._preditor.prever(self.df, sorteio=sorteio, qtd=90)
                scores_ml = {int(k): float(v) for k, v in sml.items()}
            except Exception:
                logger.exception("Ensemble: ML indisponível, a continuar sem ele")

        # Normalizo os componentes para 0–1
        def norm(series_or_dict):
            if isinstance(series_or_dict, dict):
                vals = np.array([series_or_dict.get(n, 0.0) for n in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1)], dtype=float)
            else:
                vals = series_or_dict.reindex(range(NUMERO_MINIMO, NUMERO_MAXIMO + 1)).fillna(0).to_numpy(dtype=float)
            mx = vals.max()
            if mx <= 0:
                return np.zeros_like(vals)
            return vals / mx

        f_n = norm(freq)
        a_n = norm(atr)
        m_n = norm(scores_ml)

        # Defini os pesos do ensemble: frequência 30%, atraso 20%, ML 35%, misto 15%
        # estrutura: preferir números perto da mediana histórica de posição não é trivial;
        # usamos (1 - extremos de frequência relativa) suave — aqui misturamos f e a
        scores = 0.30 * f_n + 0.20 * a_n + 0.35 * m_n + 0.15 * (0.5 * f_n + 0.5 * a_n)

        ranking = sorted(
            ((n, float(scores[n - NUMERO_MINIMO])) for n in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1)),
            key=lambda x: x[1],
            reverse=True,
        )

        # Gero opções em fatias do ranking + filtragem estrutural se possível
        ref = self._padroes_estruturais_referencia(df)
        top_pool = [n for n, _ in ranking[:40]]
        opcoes: list[list[int]] = []
        # Opções 1–N: consecutivas no ranking
        qtd = n_opcoes * QTD_NUMEROS_POR_SORTEIO
        top = [n for n, _ in ranking[:qtd]]
        for i in range(0, len(top), QTD_NUMEROS_POR_SORTEIO):
            grupo = sorted(top[i : i + QTD_NUMEROS_POR_SORTEIO])
            if len(grupo) == QTD_NUMEROS_POR_SORTEIO:
                opcoes.append(grupo)

        # Se alguma falhar os padrões, tento trocar com o pool
        rng = np.random.default_rng(seed)
        for i, op in enumerate(list(opcoes)):
            if self._respeita_padroes(op, ref):
                continue
            for _ in range(50):
                candidato = sorted(int(x) for x in rng.choice(top_pool, size=5, replace=False))
                if self._respeita_padroes(candidato, ref) and candidato not in opcoes:
                    opcoes[i] = candidato
                    break

        return {
            "opcoes": opcoes[:n_opcoes],
            "ranking_top20": ranking[:20],
            "componentes": {
                "frequencia": 0.30,
                "atraso": 0.20,
                "ml": 0.35,
                "misto": 0.15,
            },
        }

    def clustering_sorteios(
        self, n_clusters: int = 4, sorteio: Optional[str] = None
    ) -> dict:
        """Apliquei K-means exploratório à composição dos sorteios (presença 1-90).

        É curiosidade analítica — não uso isto como previsão.
        """
        df = self.df if not sorteio else self.df[self.df["Sorteio"] == sorteio]
        if len(df) < max(n_clusters * 2, 8):
            return {
                "ok": False,
                "mensagem": f"Preciso de pelo menos {max(n_clusters * 2, 8)} sorteios; tenho {len(df)}.",
                "n_clusters": n_clusters,
            }

        from sklearn.cluster import KMeans

        X = np.zeros((len(df), NUMERO_MAXIMO), dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            for c in COLUNAS_NUMEROS:
                n = int(row[c])
                if NUMERO_MINIMO <= n <= NUMERO_MAXIMO:
                    X[i, n - 1] = 1.0

        k = min(n_clusters, len(df) // 2)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        clusters = []
        for cid in range(k):
            mask = labels == cid
            membros = int(mask.sum())
            centro = km.cluster_centers_[cid]
            top_nums = sorted(
                ((i + 1, float(centro[i])) for i in range(NUMERO_MAXIMO)),
                key=lambda x: -x[1],
            )[:10]
            clusters.append({
                "id": cid,
                "tamanho": membros,
                "pct": 100.0 * membros / len(df),
                "numeros_caracteristicos": [(n, round(p, 3)) for n, p in top_nums if p > 0.05],
            })

        return {
            "ok": True,
            "n_sorteios": len(df),
            "n_clusters": k,
            "clusters": clusters,
            "mensagem": (
                "Clusters = perfis de composição no histórico. "
                "Não predizem o próximo sorteio; são só exploração."
            ),
        }

    def backtest_algoritmos(
        self,
        n_ultimos: int = 20,
        sorteio: Optional[str] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Corri um backtest rolante: para cada sorteio passado, gero 1 combinação
        com vários algoritmos usando só dados anteriores e conto os acertos.

        Incluí o baseline aleatório para uma comparação honesta.
        """
        df = self.df.sort_values(["Data", "Sorteio"]).reset_index(drop=True)
        if sorteio:
            df = df[df["Sorteio"] == sorteio].reset_index(drop=True)
        # Defini uma janela mínima de histórico antes de avaliar (independente do RF)
        min_hist = min(8, max(4, len(df) // 3))
        if len(df) < min_hist + 2:
            return pd.DataFrame()

        inicio = max(min_hist, len(df) - n_ultimos)
        rng = np.random.default_rng(seed)
        linhas = []

        algoritmos = ("frequencia", "contrarian", "estrutural", "ensemble", "aleatorio")

        for i in range(inicio, len(df)):
            hist = df.iloc[:i]
            row = df.iloc[i]
            real = set(int(row[c]) for c in COLUNAS_NUMEROS)
            data = row["Data"]
            data_s = data.strftime("%Y-%m-%d") if hasattr(data, "strftime") else str(data)[:10]
            s = row["Sorteio"]

            # Uso o histórico parcial sem mutar self: calculo os pesos no sítio
            # (não muto self)
            freq = frequencias_por_numero(hist)
            atr = atrasos_por_numero(hist)
            numeros = np.arange(NUMERO_MINIMO, NUMERO_MAXIMO + 1)

            def amostra(pesos):
                p = np.asarray(pesos, dtype=float)
                p = p / p.sum()
                esc = rng.choice(numeros, size=5, replace=False, p=p)
                return sorted(int(x) for x in esc)

            pesos_f = freq.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5
            pesos_c = atr.reindex(numeros).fillna(0).to_numpy(dtype=float) + 0.5
            pesos_u = np.ones(len(numeros), dtype=float)

            sugestoes = {
                "frequencia": amostra(pesos_f),
                "contrarian": amostra(pesos_c),
                "aleatorio": amostra(pesos_u),
            }

            # Geração estrutural
            ref = self._padroes_estruturais_referencia(hist)
            estrutural = None
            for _ in range(300):
                cand = amostra(pesos_f)
                if self._respeita_padroes(cand, ref):
                    estrutural = cand
                    break
            sugestoes["estrutural"] = estrutural or amostra(pesos_f)

            # Ensemble simples no histórico parcial: 0.5 freq + 0.3 atraso + 0.2
            f_n = pesos_f / pesos_f.max()
            a_n = pesos_c / pesos_c.max()
            score_e = 0.5 * f_n + 0.3 * a_n + 0.2
            top5 = sorted(numeros, key=lambda n: -score_e[n - NUMERO_MINIMO])[:5]
            sugestoes["ensemble"] = sorted(int(x) for x in top5)

            for nome, pred in sugestoes.items():
                acertos = len(real & set(pred))
                linhas.append({
                    "Data": data_s,
                    "Sorteio": s,
                    "Algoritmo": nome,
                    "Sugestão": ", ".join(map(str, pred)),
                    "Real": ", ".join(map(str, sorted(real))),
                    "Acertos": acertos,
                })

        return pd.DataFrame(linhas)

    def resumo_backtest_algoritmos(
        self,
        n_ultimos: int = 20,
        sorteio: Optional[str] = None,
        seed: int = 42,
    ) -> dict:
        """Agreguei as médias de acertos por algoritmo vs o baseline aleatório."""
        df = self.backtest_algoritmos(n_ultimos=n_ultimos, sorteio=sorteio, seed=seed)
        if df.empty:
            return {"ok": False, "mensagem": "Histórico insuficiente para backtest.", "medias": {}}
        medias = df.groupby("Algoritmo")["Acertos"].agg(["mean", "std", "count"])
        medias = medias.rename(columns={"mean": "media", "std": "desvio", "count": "n"})
        resultado = {
            "ok": True,
            "medias": {
                alg: {
                    "media": float(row["media"]),
                    "desvio": float(row["desvio"]) if not np.isnan(row["desvio"]) else 0.0,
                    "n": int(row["n"]),
                }
                for alg, row in medias.iterrows()
            },
            "detalhe": df,
        }
        base = resultado["medias"].get("aleatorio", {}).get("media")
        if base is not None:
            notas = []
            for alg, info in resultado["medias"].items():
                if alg == "aleatorio":
                    continue
                diff = info["media"] - base
                notas.append(
                    f"{alg}: {info['media']:.2f} (Δ vs aleatório: {diff:+.2f})"
                )
            resultado["mensagem"] = (
                "Médias de acertos (em 5 números) no backtest rolante. "
                "Se ninguém bater o aleatório de forma clara e estável, "
                "é o resultado esperado em sorteios independentes.\n\n"
                + "\n".join(notas)
            )
        return resultado
