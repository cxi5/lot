"""
lotaria_ml_streamlit.py

Interface web em Streamlit. Coloquei a lógica de dados e ML em
lotaria_core.py; aqui só trato da apresentação.

Corro com: streamlit run lotaria_ml_streamlit.py
"""

import warnings
from datetime import datetime

import pandas as pd
import streamlit as st

from lotaria_core import (
    SORTEIOS,
    COLUNAS_NUMEROS,
    NUMERO_MINIMO,
    NUMERO_MAXIMO,
    AnalisadorLotaria,
    ResultadoInvalidoError,
    criar_repositorio,
    agora_angola,
    hoje_angola,
)

# Guardei a hora real de cada sorteio em Angola
HORA_SORTEIO = {
    "Fezada": (10, 0),
    "Aqueceu": (13, 0),
    "Kazola": (16, 0),
    "Eskebra": (19, 0),
}
# Liberto as sugestões 2h30min antes (hora de Angola)
HORA_LIBERACAO = {
    "Fezada": (7, 30),
    "Aqueceu": (10, 30),
    "Kazola": (13, 30),
    "Eskebra": (16, 30),
}

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Loto · Sugestões do Dia",
    page_icon="assets/favicon.png",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ===== Identidade visual: branco / azul / vermelho ===== */
    .bola-numero {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.7rem;
        height: 2.7rem;
        border-radius: 50%;
        background: linear-gradient(145deg, #0a4d8c 0%, #06315c 100%);
        color: #ffffff;
        font-weight: 700;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 1.05rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        box-shadow: 0 3px 8px rgba(10, 77, 140, 0.35);
    }
    .bola-numero.destaque {
        background: linear-gradient(145deg, #c8102e 0%, #9a0c24 100%);
    }
    .linha-sorteio { margin-bottom: 0.7rem; }
    .linha-sorteio .rotulo {
        color: #0a4d8c;
        font-size: 0.88rem;
        font-weight: 600;
        display: block;
        margin-bottom: 0.25rem;
    }
    .celula-grelha {
        text-align: center;
        border-radius: 10px;
        padding: 0.5rem 0.2rem;
        margin-bottom: 0.4rem;
        font-family: 'Segoe UI', system-ui, sans-serif;
        background: #e8f1fa;
        border: 1px solid #d0dce8;
    }
    .celula-grelha .numero { font-size: 1.05rem; font-weight: 700; color: #0a4d8c; }
    .celula-grelha .meta { font-size: 0.62rem; opacity: 0.85; color: #555; }

    /* Cards / boxes estilo sugestões */
    .sugestao-card {
        background: #ffffff;
        border-radius: 14px;
        border-top: 4px solid #c8102e;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(10, 77, 140, 0.1);
    }
    .sugestao-card h3 {
        color: #0a4d8c;
        margin: 0 0 0.4rem 0;
        font-size: 1.15rem;
    }
    .aviso-box {
        background: #fff5f5;
        border-left: 4px solid #c8102e;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #5c1a1a;
        margin-bottom: 1rem;
    }
    .horario-badge {
        background: #e8f1fa;
        color: #0a4d8c;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    /* Sidebar: fundo azul, texto claro nos labels — mas NÃO nos inputs */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a4d8c 0%, #06315c 100%);
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stRadio label {
        color: #e8f1fa !important;
    }
    /* Inputs legíveis (senha, números, select, date) */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"],
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] input::placeholder {
        color: #888 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def bolas_html(numeros: list[int]) -> str:
    """Montei as bolas HTML com os números."""
    return "".join(f'<span class="bola-numero">{n}</span>' for n in numeros)


def linha_sorteio(rotulo: str, numeros: list[int]) -> None:
    st.markdown(
        f'<div class="linha-sorteio"><span class="rotulo">{rotulo}</span>{bolas_html(numeros)}</div>',
        unsafe_allow_html=True,
    )


def linha_tres_opcoes(rotulo: str, opcoes: list[list[int]]) -> None:
    """Mostrei o rótulo e as N opções de 5 números."""
    st.markdown(f"**{rotulo}**")
    for i, nums in enumerate(opcoes, start=1):
        st.markdown(
            f'<div class="linha-sorteio"><span class="rotulo">Opção {i}</span>{bolas_html(nums)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("")


def texto_partilha(tres_opcoes: dict) -> str:
    """Montei este texto para copiar no WhatsApp."""
    linhas = ["🎯 Loto · Sugestões do Dia", ""]
    for sorteio, ops in tres_opcoes.items():
        linhas.append(f"*{sorteio}*")
        for i, nums in enumerate(ops, 1):
            linhas.append(f"  Opção {i}: {', '.join(map(str, nums))}")
        linhas.append("")
    linhas.append("_App Loto_")
    return "\n".join(linhas)


def badge_liberacao(sorteio: str, minutos_atual: int) -> str:
    h, m = HORA_LIBERACAO.get(sorteio, (0, 0))
    if minutos_atual >= h * 60 + m:
        return f"🟢 **{sorteio}** · liberado"
    return f"⚪ **{sorteio}** · a partir das {h:02d}:{m:02d}"


@st.cache_resource
def get_analyzer() -> AnalisadorLotaria:
    """Montei o repositório conforme os secrets.
    Sem secrets uso CSV local; na Cloud uso Supabase se estiver definido.
    """
    backend = st.secrets.get("BACKEND", "csv")
    try:
        repositorio = criar_repositorio(
            backend=backend,
            supabase_url=st.secrets.get("SUPABASE_URL"),
            supabase_key=st.secrets.get("SUPABASE_KEY"),
        )
    except ValueError as erro:
        st.error(
            f"Configuração de backend inválida: {erro}\n\n"
            "Confirma BACKEND, SUPABASE_URL e SUPABASE_KEY em .streamlit/secrets.toml."
        )
        st.stop()
    return AnalisadorLotaria(repositorio=repositorio)


def verificar_senha() -> bool:
    """Peço senha só se APP_PASSWORD existir nos secrets.
    A senha fica só no secrets.toml / painel do Cloud, nunca no código.
    """
    senha_configurada = st.secrets.get("APP_PASSWORD")
    if not senha_configurada:
        return True
    if st.session_state.get("autenticado"):
        return True

    with st.sidebar.form("login_form"):
        st.write("Login necessario para adicionar resultados")
        senha = st.text_input("Senha", type="password")
        enviado = st.form_submit_button("Entrar")

    if enviado:
        if senha == senha_configurada:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.sidebar.error("Senha incorreta.")

    return False


def _extrair_numeros_ocr(imagem_bytes: bytes) -> list[int]:
    """Tirei até 5 números (1-90) da foto do boletim com OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io
        import re

        img = Image.open(io.BytesIO(imagem_bytes))
        # Escala de cinza melhora o OCR em muitos casos
        img = img.convert("L")
        texto = pytesseract.image_to_string(img, lang="por+eng", config="--psm 6")
        # Encontra todos os números de 1 ou 2 dígitos entre 1 e 90
        candidatos = re.findall(r"\b([1-9]|[1-8][0-9]|90)\b", texto)
        numeros: list[int] = []
        for c in candidatos:
            n = int(c)
            if NUMERO_MINIMO <= n <= NUMERO_MAXIMO and n not in numeros:
                numeros.append(n)
            if len(numeros) >= 5:
                break
        return numeros
    except Exception as e:
        # Qualquer falha (tesseract em falta, imagem inválida, etc.)
        return []


def secao_adicionar_resultado(analyzer: AnalisadorLotaria) -> None:
    st.sidebar.header("Adicionar Resultado")
    st.sidebar.caption(f"Hora em Angola: {agora_angola().strftime('%d/%m/%Y %H:%M')}")
    data_input = st.sidebar.date_input("Data", value=agora_angola().date())
    sorteio_input = st.sidebar.selectbox("Sorteio", SORTEIOS)

    # --- OCR de foto do boletim ---
    st.sidebar.markdown("**Ou fotografa o boletim**")
    foto = st.sidebar.file_uploader(
        "Foto do boletim / resultado",
        type=["png", "jpg", "jpeg", "webp"],
        key="ocr_uploader",
        help="Tira uma foto nítida dos 5 números. O OCR tenta preenchê-los automaticamente.",
    )

    valores_default = [i + 1 for i in range(len(COLUNAS_NUMEROS))]
    if foto is not None:
        with st.sidebar.spinner("A ler números da foto..."):
            extraidos = _extrair_numeros_ocr(foto.getvalue())
        if len(extraidos) >= 5:
            valores_default = extraidos[:5]
            st.sidebar.success(f"OCR encontrou: {', '.join(map(str, valores_default))}")
        elif extraidos:
            valores_default = (extraidos + valores_default)[:5]
            st.sidebar.warning(
                f"OCR encontrou apenas {len(extraidos)} número(s): "
                f"{', '.join(map(str, extraidos))}. Confirma e completa manualmente."
            )
        else:
            st.sidebar.warning(
                "OCR não conseguiu ler números. Verifica a nitidez da foto "
                "ou preenche manualmente. (Em ambiente local/Termux é preciso "
                "ter o Tesseract instalado.)"
            )

    data_str_preview = data_input.strftime("%Y-%m-%d")
    existente = analyzer.obter_resultado(data_str_preview, sorteio_input)
    if existente is not None:
        hs, ms = HORA_SORTEIO.get(sorteio_input, (0, 0))
        st.sidebar.warning(
            f"Já existe registo para **{data_str_preview} · {sorteio_input}** "
            f"({hs:02d}:{ms:02d}):\n"
            f"{', '.join(map(str, existente))}\n\n"
            "Se gravar números **diferentes**, o anterior é substituído."
        )
        # Pré-preencho com o existente se o utilizador não veio do OCR
        if foto is None:
            valores_default = list(existente)

    numeros = [
        st.sidebar.number_input(
            f"Número {i + 1}",
            min_value=NUMERO_MINIMO,
            max_value=NUMERO_MAXIMO,
            value=valores_default[i],
            key=f"num_input_{i}",
        )
        for i in range(len(COLUNAS_NUMEROS))
    ]

    confirmar_subst = False
    try:
        conflito = analyzer.verificar_conflito_registo(
            data_str_preview, sorteio_input, list(numeros)
        )
    except ResultadoInvalidoError:
        conflito = None

    if conflito and conflito["status"] == "diferente":
        st.sidebar.error(
            "⚠️ **Registo duplicado com números diferentes**\n\n"
            f"Data: {data_str_preview} · {sorteio_input}\n"
            f"Actual: {', '.join(map(str, conflito['existente']))}\n"
            f"Novo: {', '.join(map(str, conflito['novos']))}"
        )
        confirmar_subst = st.sidebar.checkbox(
            "Confirmo substituir o registo anterior",
            value=False,
            key="confirmar_substituicao",
        )
    elif conflito and conflito["status"] == "igual":
        st.sidebar.info("Estes números já estão gravados — não há alterações.")

    if st.sidebar.button("Salvar Resultado"):
        try:
            data_str = data_input.strftime("%Y-%m-%d")
            conflito_save = analyzer.verificar_conflito_registo(
                data_str, sorteio_input, list(numeros)
            )
            if conflito_save["status"] == "diferente" and not confirmar_subst:
                st.sidebar.error(
                    "Marca a caixa **Confirmo substituir** para gravar números diferentes "
                    "no mesmo data/sorteio."
                )
                st.stop()
            if conflito_save["status"] == "igual":
                st.sidebar.info("Nada a gravar — resultado idêntico ao existente.")
                st.stop()
            analyzer.adicionar_resultado(data_str, sorteio_input, list(numeros))
        except ResultadoInvalidoError as erro:
            st.sidebar.error(str(erro))
        else:
            # Dou feedback imediato dos acertos face às sugestões congeladas
            confronto = analyzer.comparar_resultados_com_sugestoes(data=data_str)
            info = confronto.get(sorteio_input)
            if info and info.get("acertos"):
                linhas_hit = []
                for i, (ac, nums_hit) in enumerate(
                    zip(info["acertos"], info["numeros_acertados"]), 1
                ):
                    if ac > 0:
                        linhas_hit.append(
                            f"Opção {i}: {ac} acerto(s) "
                            f"({', '.join(map(str, nums_hit))})"
                        )
                resumo = (
                    f"Melhor: opção #{info['melhor_opcao']} "
                    f"com {info['melhor']} acerto(s)."
                )
                if linhas_hit:
                    st.sidebar.success(
                        f"Resultado salvo · **Sugestões acertadas · {sorteio_input}**\n\n"
                        + "\n".join(linhas_hit)
                        + f"\n\n{resumo}"
                    )
                else:
                    st.sidebar.success(
                        f"Resultado salvo · {sorteio_input}: 0 acertos nas 4 opções. "
                        "O modelo vai aprender com estes números."
                    )
            else:
                st.sidebar.success(
                    "Resultado salvo! O modelo vai aprender com estes números "
                    "nas próximas sugestões."
                )
            st.cache_resource.clear()
            st.rerun()


def pagina_dashboard(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Sugestões do Dia")
    agora = agora_angola()
    hoje = hoje_angola()
    st.caption(
        f"4 opções por sorteio · agora em Angola: **{agora.strftime('%d/%m/%Y %H:%M')}** (WAT)"
    )

    if analyzer.df.empty:
        st.info(
            "Ainda não há resultados na base. "
            "**Adiciona o primeiro resultado** na barra lateral (data, sorteio e 5 números) "
            "para o modelo começar a trabalhar."
        )
        return

    minutos_atual = agora.hour * 60 + agora.minute

    # Mostro os badges de liberação
    cols_b = st.columns(4)
    for i, sorteio in enumerate(SORTEIOS):
        with cols_b[i]:
            st.markdown(badge_liberacao(sorteio, minutos_atual))

    with st.spinner("A calcular 4 opções por sorteio..."):
        tres_opcoes = analyzer.prever_tres_opcoes_por_sorteio()

    liberados = [
        s for s in SORTEIOS
        if minutos_atual >= HORA_LIBERACAO[s][0] * 60 + HORA_LIBERACAO[s][1]
    ]

    if not liberados:
        st.warning("Nenhuma sugestão liberada ainda. Fezada abre às 07:30 (hora de Angola).")
    else:
        for sorteio in liberados:
            hs, ms = HORA_SORTEIO[sorteio]
            rotulo = f"{sorteio} — sorteio às {hs:02d}:{ms:02d}"
            linha_tres_opcoes(rotulo, tres_opcoes[sorteio])

        texto = texto_partilha({s: tres_opcoes[s] for s in liberados})
        st.text_area("Copiar para WhatsApp / partilhar", value=texto, height=180)
        st.download_button(
            "Descarregar sugestões (.txt)",
            data=texto.encode("utf-8"),
            file_name="sugestoes_do_dia.txt",
            mime="text/plain",
        )

    # Secção: resultados vencedores de hoje
    st.divider()
    st.subheader(f"Resultados vencedores · {hoje}")
    df_hoje = analyzer.df[analyzer.df["Data"].dt.strftime("%Y-%m-%d") == hoje]
    if df_hoje.empty:
        st.caption("Ainda não gravei nenhum resultado de hoje. Usa a barra lateral para adicionar.")
    else:
        confronto = analyzer.resultado_vs_sugestoes_hoje()
        for _, row in df_hoje.sort_values("Sorteio").iterrows():
            s = row["Sorteio"]
            hs, ms = HORA_SORTEIO.get(s, (0, 0))
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            st.markdown(f"**{s}** · sorteio {hs:02d}:{ms:02d} · {hoje}")
            st.markdown(bolas_html(nums), unsafe_allow_html=True)
            if s in confronto:
                info = confronto[s]
                st.markdown(f"**Sugestões acertadas · {s}**")
                for i, (op, ac) in enumerate(zip(info["opcoes"], info["acertos"]), 1):
                    nums_hit = info.get("numeros_acertados", [[]] * len(info["opcoes"]))[i - 1]
                    hit_txt = (
                        f" → acertou: {', '.join(map(str, nums_hit))}"
                        if nums_hit else ""
                    )
                    destaque = " ⭐" if i == info.get("melhor_opcao") else ""
                    st.caption(
                        f"Opção {i}: {', '.join(map(str, op))} → "
                        f"**{ac} acerto(s)**{hit_txt}{destaque}"
                    )
                st.markdown(
                    f"Melhor opção: **#{info.get('melhor_opcao', '—')}** "
                    f"com **{info['melhor']}** número(s) certos."
                )

    # Últimos resultados vencedores (dias anteriores)
    st.divider()
    st.subheader("Últimos resultados vencedores")
    ultimos = analyzer.df.sort_values("Data", ascending=False).head(8)
    for _, row in ultimos.iterrows():
        s = row["Sorteio"]
        hs, ms = HORA_SORTEIO.get(s, (0, 0))
        data = row["Data"]
        data_s = data.strftime("%d/%m/%Y") if hasattr(data, "strftime") else str(data)[:10]
        nums = [int(row[c]) for c in COLUNAS_NUMEROS]
        st.markdown(
            f"**{data_s}** · {s} ({hs:02d}:{ms:02d}) — "
            + ", ".join(map(str, nums))
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total de sorteios na base", len(analyzer.df))
    with c2:
        freq_geral = analyzer.frequencias()
        mais_frequente = freq_geral.idxmax() if freq_geral.sum() else "-"
        st.metric("Número mais frequente (geral)", mais_frequente)


def pagina_analise_por_sorteio(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Análise por Sorteio")
    if analyzer.df.empty:
        st.info("Sem dados ainda. Adiciona resultados na barra lateral.")
        return

    sorteio_sel = st.selectbox("Escolha o sorteio", SORTEIOS)

    freq = analyzer.frequencias(sorteio_sel)
    atraso = analyzer.atrasos(sorteio_sel)
    df_s = analyzer.df[analyzer.df["Sorteio"] == sorteio_sel]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Frequência", "Atraso", "Últimos resultados", "Pares e trios"]
    )
    with tab1:
        st.bar_chart(freq)
        st.caption(f"Aparições de cada número no histórico de {sorteio_sel}")
    with tab2:
        st.bar_chart(atraso)
        st.caption("Sorteios desde a última aparição de cada número")
    with tab3:
        ultimos = df_s.sort_values("Data", ascending=False).head(15)
        st.dataframe(ultimos, use_container_width=True, hide_index=True)
    with tab4:
        st.markdown("**Pares que mais saem juntos**")
        pares = analyzer.pares_frequentes(sorteio_sel, tamanho=2, top=10)
        if not pares:
            st.caption("Dados insuficientes.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [{"Par": f"{a} · {b}", "Vezes": n} for (a, b), n in pares]
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.markdown("**Trios que mais saem juntos**")
        trios = analyzer.pares_frequentes(sorteio_sel, tamanho=3, top=10)
        if not trios:
            st.caption("Dados insuficientes.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [{"Trio": f"{a} · {b} · {c}", "Vezes": n} for (a, b, c), n in trios]
                ),
                hide_index=True,
                use_container_width=True,
            )


def pagina_grelha(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Grelha completa 1-90")
    st.caption("Intensidade de cor = frequência histórica · texto pequeno = atraso actual")
    if analyzer.df.empty:
        st.info("Sem dados ainda. Adiciona resultados na barra lateral.")
        return

    freq = analyzer.frequencias()
    atraso = analyzer.atrasos()
    max_freq = max(int(freq.max()), 1)

    cols = st.columns(10)
    for numero in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
        f = int(freq[numero])
        a = int(atraso[numero])
        intensidade = f / max_freq
        # Apliquei o gradiente do feltro (frio) ao latão (quente), no tema do app
        if intensidade > 0.7:
            fundo, cor_texto = "#C9A227", "#152019"
        elif intensidade > 0.45:
            fundo, cor_texto = "#8C6D14", "#EDE6D6"
        elif intensidade > 0.2:
            fundo, cor_texto = "#4A5C4F", "#EDE6D6"
        else:
            fundo, cor_texto = "#1F2E24", "#EDE6D6"

        with cols[(numero - 1) % 10]:
            st.markdown(
                f"""
                <div class="celula-grelha" style="background:{fundo}; color:{cor_texto};">
                    <div class="numero">{numero}</div>
                    <div class="meta">{f}x \u00b7 atr {a}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pagina_historico(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Histórico de resultados vencedores")
    st.caption(f"Hora actual em Angola: {agora_angola().strftime('%d/%m/%Y %H:%M')} (WAT)")

    if analyzer.df.empty:
        st.info("Ainda não há resultados. Adiciona na barra lateral ou importa um CSV abaixo.")
    else:
        # --- Qualidade dos dados ---
        with st.expander("🔍 Qualidade dos dados", expanded=False):
            rel = analyzer.relatorio_qualidade()
            r = rel["resumo"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registos", rel["n_registos"])
            c2.metric("Linhas inválidas", r["invalidas"])
            c3.metric("Duplicados (data+sorteio)", r["duplicados"])
            c4.metric("Sorteios em falta", r["sorteios_em_falta"])

            if r["invalidas"] == 0 and r["duplicados"] == 0 and r["sorteios_em_falta"] == 0:
                st.success("Base limpa: sem inválidos, duplicados ou sorteios em falta no período.")
            else:
                if rel["linhas_invalidas"]:
                    st.markdown("**Linhas com erros de digitação / validação**")
                    for item in rel["linhas_invalidas"]:
                        st.error(
                            f"{item['data']} · {item['sorteio']} → "
                            f"{', '.join(map(str, item['numeros']))} — "
                            + "; ".join(item["problemas"])
                        )
                if rel["duplicados_chave"]:
                    st.markdown("**Registos duplicados (mesma data + sorteio)**")
                    for item in rel["duplicados_chave"]:
                        if item["numeros_diferentes"]:
                            st.error(
                                f"{item['data']} · {item['sorteio']}: "
                                f"{item['ocorrencias']} ocorrências com números **diferentes**"
                            )
                            for i, v in enumerate(item["variantes"], 1):
                                st.caption(f"  Variante {i}: {', '.join(map(str, v))}")
                        else:
                            st.warning(
                                f"{item['data']} · {item['sorteio']}: "
                                f"{item['ocorrencias']} ocorrências idênticas"
                            )
                if rel["datas_incompletas"]:
                    st.markdown("**Dias com sorteios em falta**")
                    for item in rel["datas_incompletas"][:40]:
                        falta = ", ".join(
                            f"{s} ({next((x['hora'] for x in rel['sorteios_em_falta'] if x['data']==item['data'] and x['sorteio']==s), '')})"
                            for s in item["em_falta"]
                        )
                        st.caption(
                            f"{item['data']}: {item['n_presentes']}/4 — falta: {falta}"
                        )
                    if len(rel["datas_incompletas"]) > 40:
                        st.caption(f"… e mais {len(rel['datas_incompletas']) - 40} dias.")

        filtro = st.multiselect("Filtrar por sorteio", SORTEIOS, default=SORTEIOS)
        df_filtrado = analyzer.df[analyzer.df["Sorteio"].isin(filtro)].sort_values(
            "Data", ascending=False
        ).copy()
        # Mostrei a data legível e a hora real do sorteio
        df_view = df_filtrado.copy()
        df_view["Data"] = df_view["Data"].dt.strftime("%d/%m/%Y")
        df_view["Hora"] = df_view["Sorteio"].map(
            lambda s: f"{HORA_SORTEIO[s][0]:02d}:{HORA_SORTEIO[s][1]:02d}"
            if s in HORA_SORTEIO else ""
        )
        cols_ordem = ["Data", "Hora", "Sorteio"] + COLUNAS_NUMEROS
        st.dataframe(df_view[cols_ordem], use_container_width=True, height=400)
        st.download_button(
            label="Baixar CSV",
            data=analyzer.df.to_csv(index=False).encode("utf-8"),
            file_name="resultados_lotaria_angola.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### Corrigir resultado errado")
        st.caption(
            "Escolhe a data e o sorteio do registo errado. "
            "Podes remover ou gravar de novo com os números certos (substitui o anterior)."
        )
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            datas = sorted(
                {d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                 for d in analyzer.df["Data"]},
                reverse=True,
            )
            data_rm = st.selectbox("Data", datas, key="rm_data")
        with c2:
            sorteio_rm = st.selectbox("Sorteio", SORTEIOS, key="rm_sorteio")
        with c3:
            st.write("")
            st.write("")
            if st.button("Remover", type="primary"):
                if analyzer.remover_resultado(data_rm, sorteio_rm):
                    st.success(f"Removido: {data_rm} · {sorteio_rm}")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("Não encontrei esse registo.")

        # Mostro o registo seleccionado, se existir
        data_ts = pd.to_datetime(data_rm)
        match = analyzer.df[
            (analyzer.df["Data"] == data_ts) & (analyzer.df["Sorteio"] == sorteio_rm)
        ]
        if not match.empty:
            row = match.iloc[0]
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            hs, ms = HORA_SORTEIO.get(sorteio_rm, (0, 0))
            st.info(
                f"Registo actual: **{data_rm}** · {sorteio_rm} ({hs:02d}:{ms:02d}) → "
                + ", ".join(map(str, nums))
            )
            st.caption(
                "Para corrigir: na barra lateral mete a mesma data e sorteio com os números certos e grava — substitui este."
            )

    st.divider()
    st.markdown("### Importar CSV em massa")
    st.caption(
        "Colunas obrigatórias: `Data`, `Sorteio`, `N1`, `N2`, `N3`, `N4`, `N5`. "
        "Linhas inválidas são ignoradas."
    )
    ficheiro = st.file_uploader("Ficheiro CSV", type=["csv"], key="import_csv")
    if ficheiro is not None:
        try:
            df_imp = pd.read_csv(ficheiro)
            st.dataframe(df_imp.head(10), use_container_width=True)
            if st.button("Importar para a base"):
                n = analyzer.importar_dataframe(df_imp)
                st.success(f"Importados **{n}** resultados. O modelo vai aprender com eles.")
                st.cache_resource.clear()
                st.rerun()
        except Exception as exc:
            st.error(f"Não consegui ler o CSV: {exc}")


def pagina_modelo_ml(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Modelo Random Forest")
    st.caption(
        "O modelo é treinado automaticamente sempre que adicionas um resultado. "
        "Aqui podes forçar retreino, ver o erro (MAE) e a importância das features."
    )

    sorteio_sel = st.selectbox("Sorteio a avaliar", ["Geral (todos)"] + SORTEIOS)
    sorteio_arg = None if sorteio_sel == "Geral (todos)" else sorteio_sel

    df_local = analyzer.df if sorteio_arg is None else analyzer.df[analyzer.df["Sorteio"] == sorteio_arg]
    n_hist = len(df_local)
    minimo = analyzer.config.minimo_sorteios_para_treino
    c_stat1, c_stat2, c_stat3 = st.columns(3)
    with c_stat1:
        st.metric("Sorteios no histórico", n_hist)
    with c_stat2:
        st.metric("Mínimo para treinar", minimo)
    with c_stat3:
        status = "Pronto a treinar" if n_hist >= minimo else "Histórico insuficiente"
        st.metric("Estado", status)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Retreinar sob demanda")
        st.caption(
            "O modelo já é retreinado automaticamente sempre que um "
            "resultado novo é adicionado. Usa este botão só se quiseres "
            "forçar um retreino manual."
        )
        if st.button("Treinar agora", type="primary"):
            with st.spinner("A treinar Random Forest..."):
                modelo = analyzer.retreinar(sorteio_arg)
            if modelo is None:
                st.warning(
                    f"Histórico insuficiente para treinar ({n_hist}/{minimo} sorteios). "
                    "Adiciona mais resultados."
                )
            else:
                st.success(
                    f"Modelo treinado com sucesso "
                    f"({analyzer.config.n_estimators} árvores, "
                    f"max_depth={analyzer.config.max_depth})."
                )

    with col_b:
        st.markdown("#### Avaliação (erro do modelo)")
        st.caption(
            "MAE calculado com split temporal (sem embaralhar): "
            "treina nos primeiros 80% e mede o erro nos últimos 20% "
            "— evita vazamento de informação futura."
        )
        with st.spinner("A avaliar..."):
            resultado = analyzer.avaliar_modelo(sorteio_arg)
        if resultado.get("mae") is None:
            st.warning(
                f"Amostras insuficientes ({resultado.get('n_amostras', 0)}) para avaliar."
            )
        else:
            st.metric("MAE (erro médio absoluto)", f"{resultado['mae']:.4f}")
            st.caption(f"Calculado sobre {resultado['n_amostras']} exemplos")

    st.divider()
    st.markdown("#### Importância das features")
    st.caption(
        "Mostra o que o modelo está a usar de facto: frequência total, "
        "frequência recente, atraso, peso temporal, posição média, par/ímpar."
    )
    importancias = analyzer.importancia_features(sorteio_arg)
    if importancias is None:
        st.info(
            "Treina o modelo (botão acima ou gera uma previsão no Dashboard) "
            "para ver a importância das features."
        )
    else:
        st.bar_chart(importancias)

    st.divider()
    st.markdown(
        f"""
        **Parâmetros atuais do Random Forest**
        - `n_estimators` = {analyzer.config.n_estimators}
        - `max_depth` = {analyzer.config.max_depth}
        - `minimo_sorteios_para_treino` = {analyzer.config.minimo_sorteios_para_treino}
        - `janela_recente` = {analyzer.config.janela_recente}
        - Combinação final do ranking: **65% ML + 25% frequência + 10% atraso**
        """
    )

    st.divider()
    st.markdown("#### Backtest das sugestões")
    st.caption(
        "Para cada sorteio passado, o modelo treina só com dados anteriores e "
        "compara a 1.ª sugestão com o resultado real."
    )
    if analyzer.df.empty:
        st.info("Precisas de histórico para correr o backtest.")
    else:
        c_bt1, c_bt2 = st.columns(2)
        with c_bt1:
            n_bt = st.slider("Últimos N sorteios", min_value=5, max_value=40, value=15)
        with c_bt2:
            s_bt = st.selectbox(
                "Filtrar sorteio (backtest)",
                ["Todos"] + SORTEIOS,
                key="bt_sorteio",
            )
        if st.button("Correr backtest", key="btn_backtest"):
            with st.spinner("A simular sugestões no passado..."):
                df_bt = analyzer.backtest_sugestoes(
                    n_ultimos=n_bt,
                    sorteio=None if s_bt == "Todos" else s_bt,
                )
            if df_bt.empty:
                st.warning(
                    "Não foi possível gerar o backtest (histórico curto ou abaixo do mínimo de treino)."
                )
            else:
                media = float(df_bt["Acertos"].mean())
                st.metric("Média de acertos (em 5 números)", f"{media:.2f}")
                st.dataframe(df_bt, use_container_width=True, hide_index=True)
                dist = df_bt["Acertos"].value_counts().sort_index()
                st.bar_chart(dist)
                st.caption("Distribuição de acertos nos sorteios testados.")


def pagina_verificar_numeros(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Verificar os meus números")
    st.caption(
        "Mete os 5 números do teu bilhete. Vês frequência, atraso, quantas vezes "
        "saíram juntos e quantos coincidem com as sugestões de hoje."
    )

    sorteio_v = st.selectbox(
        "Sorteio de referência (opcional)",
        ["Geral (todos)"] + SORTEIOS,
        key="ver_sorteio",
    )
    cols = st.columns(5)
    nums = []
    for i in range(5):
        with cols[i]:
            nums.append(
                st.number_input(
                    f"N{i + 1}",
                    min_value=NUMERO_MINIMO,
                    max_value=NUMERO_MAXIMO,
                    value=i + 1,
                    key=f"ver_n_{i}",
                )
            )

    if st.button("Analisar bilhete", type="primary", key="btn_ver_bilhete"):
        try:
            info = analyzer.analisar_bilhete(
                list(nums),
                None if sorteio_v == "Geral (todos)" else sorteio_v,
            )
        except ResultadoInvalidoError as err:
            st.error(str(err))
            return

        st.markdown("**Os teus números**")
        st.markdown(bolas_html(info["numeros"]), unsafe_allow_html=True)

        st.markdown("**Frequência e atraso de cada um**")
        st.dataframe(
            pd.DataFrame(info["detalhe"]).rename(
                columns={
                    "numero": "Número",
                    "frequencia": "Frequência",
                    "atraso": "Atraso",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        if info["vezes_juntos"] > 0:
            st.success(
                f"Estes 5 números saíram **juntos** {info['vezes_juntos']} vez(es). "
                f"Última vez: **{info['ultima_vez_juntos']}**."
            )
        else:
            st.info("Estes 5 números nunca saíram todos juntos no histórico desta base.")

        if info["overlap_sugestoes_hoje"]:
            st.markdown("**Coincidências com as sugestões de hoje**")
            for rotulo, comuns in info["overlap_sugestoes_hoje"].items():
                st.write(f"- {rotulo}: {', '.join(map(str, comuns))}")
        else:
            st.caption("Nenhuma coincidência com as 4 opções de hoje (ou ainda sem dados para prever).")




def pagina_historico_sugestoes(analyzer: AnalisadorLotaria) -> None:
    """Mostro o histórico das sugestões por dia/sorteio, com números reais e acertos."""
    st.subheader("Histórico de sugestões")
    st.caption(
        "Sugestões congeladas por data e sorteio. Quando o resultado real é lançado, "
        "mostra quantos números cada opção acertou."
    )

    stats = analyzer.estatisticas_acertos()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sorteios avaliados", stats["n_sorteios"])
    with c2:
        media = stats["media_melhor"]
        st.metric(
            "Média (melhor opção)",
            f"{media:.2f}" if media is not None else "—",
        )
    with c3:
        media_t = stats["media_todas_opcoes"]
        st.metric(
            "Média (todas as 4 opções)",
            f"{media_t:.2f}" if media_t is not None else "—",
        )

    if stats["por_sorteio"]:
        st.markdown("**Média por sorteio (melhor opção)**")
        cols = st.columns(len(stats["por_sorteio"]))
        for col, (s, info) in zip(cols, stats["por_sorteio"].items()):
            with col:
                st.metric(s, f"{info['media_melhor']:.2f}", help=f"{info['n']} sorteios")

    if stats["por_dia"]:
        with st.expander("Média diária de acertos (melhor opção)"):
            linhas = []
            for d, info in list(stats["por_dia"].items())[:30]:
                linhas.append(
                    {
                        "Data": d,
                        "Sorteios": info["n"],
                        "Média melhor opção": round(info["media_melhor"], 2),
                    }
                )
            st.dataframe(linhas, use_container_width=True, hide_index=True)

    st.divider()

    hist = analyzer.historico_sugestoes()
    if not hist:
        st.info(
            "Ainda não há sugestões congeladas. Abre o **Dashboard** para gerar "
            "as de hoje — ficam guardadas automaticamente."
        )
        return

    filtro = st.multiselect(
        "Filtrar por sorteio", SORTEIOS, default=SORTEIOS, key="hist_sug_filtro"
    )
    hist = [h for h in hist if h["sorteio"] in filtro]

    so_com_resultado = st.checkbox(
        "Só mostrar sorteios com resultado real já lançado",
        value=False,
        key="hist_sug_so_real",
    )
    if so_com_resultado:
        hist = [h for h in hist if h["real"] is not None]

    if not hist:
        st.warning("Nenhum registo com os filtros actuais.")
        return

    for item in hist:
        data_legivel = item["data"]
        try:
            from datetime import datetime as _dt
            data_legivel = _dt.strptime(item["data"], "%Y-%m-%d").strftime("%d/%m/%Y")
        except Exception:
            pass

        titulo = (
            f"**{data_legivel}** · {item['sorteio']} "
            f"(sorteio {item['hora_sorteio']})"
        )
        if item["bloqueado_em"]:
            titulo += f" · sugerido às {item['bloqueado_em'][11:16] if len(item['bloqueado_em']) >= 16 else item['bloqueado_em']}"

        with st.expander(titulo, expanded=(item["data"] == hoje_angola())):
            if item["real"] is not None:
                st.markdown("**Números vencedores**")
                st.markdown(bolas_html(item["real"]), unsafe_allow_html=True)
                st.markdown(
                    f"**Sugestões acertadas · {item['sorteio']}** — "
                    f"melhor opção **#{item['melhor_opcao']}** "
                    f"({item['melhor']} acerto(s))"
                )
                for i, (op, ac) in enumerate(zip(item["opcoes"], item["acertos"]), 1):
                    nums_hit = item["numeros_acertados"][i - 1]
                    hit_txt = (
                        f" · acertou: {', '.join(map(str, nums_hit))}"
                        if nums_hit else ""
                    )
                    star = " ⭐" if i == item["melhor_opcao"] else ""
                    st.caption(
                        f"Opção {i}: {', '.join(map(str, op))} → "
                        f"**{ac} acerto(s)**{hit_txt}{star}"
                    )
            else:
                st.caption("Resultado real ainda não lançado.")
                for i, op in enumerate(item["opcoes"], 1):
                    st.caption(f"Opção {i}: {', '.join(map(str, op))}")


def pagina_algoritmos(analyzer: AnalisadorLotaria) -> None:
    """Expus geradores alternativos, ensemble, clustering e backtest vs aleatório."""
    st.subheader("Algoritmos de sugestões")
    st.caption(
        "Experiências: amostragem ponderada, filtragem estrutural, ensemble, "
        "clustering e backtest rolante comparado com escolha aleatória. "
        "Sorteios são independentes — nenhum método garante acertos."
    )
    if analyzer.df.empty:
        st.info("Sem dados. Adiciona resultados primeiro.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        sorteio_f = st.selectbox("Sorteio", ["Todos"] + SORTEIOS, key="alg_sorteio")
    with c2:
        n_ops = st.slider("Nº de opções", 1, 6, 4, key="alg_nops")
    with c3:
        seed = st.number_input("Seed (reprodutibilidade)", min_value=0, value=42, key="alg_seed")
    sorteio = None if sorteio_f == "Todos" else sorteio_f

    tab1, tab2, tab3, tab4 = st.tabs([
        "Gerar sugestões",
        "Ensemble",
        "Clustering",
        "Backtest vs aleatório",
    ])

    with tab1:
        st.markdown("**Amostragem ponderada e filtragem estrutural**")
        modo = st.radio(
            "Modo",
            [
                "frequencia — peso ∝ frequência (quentes)",
                "contrarian — peso ∝ atraso (frios)",
                "estrutural — só combinações com soma/paridade/consecutivos típicos",
                "uniforme — baseline aleatório",
            ],
            key="alg_modo",
        )
        if st.button("Gerar", key="alg_gerar"):
            if modo.startswith("frequencia"):
                ops = analyzer.gerar_amostra_ponderada(
                    sorteio=sorteio, n_opcoes=n_ops, modo="frequencia", seed=int(seed)
                )
            elif modo.startswith("contrarian"):
                ops = analyzer.gerar_amostra_ponderada(
                    sorteio=sorteio, n_opcoes=n_ops, modo="contrarian", seed=int(seed)
                )
            elif modo.startswith("uniforme"):
                ops = analyzer.gerar_amostra_ponderada(
                    sorteio=sorteio, n_opcoes=n_ops, modo="uniforme", seed=int(seed)
                )
            else:
                ops = analyzer.gerar_filtragem_estrutural(
                    sorteio=sorteio, n_opcoes=n_ops, seed=int(seed)
                )
            if not ops:
                st.warning("Não consegui gerar opções com estes filtros.")
            else:
                for i, op in enumerate(ops, 1):
                    st.markdown(f"**Opção {i}**")
                    st.markdown(bolas_html(op), unsafe_allow_html=True)
                    av = analyzer.avaliar_soma_combinacao(op)
                    st.caption(av["mensagem"])
            st.info(
                "Isto é experimental. Não substitui as sugestões congeladas do Dashboard "
                "nem aumenta a probabilidade real de ganhar."
            )

    with tab2:
        st.markdown(
            "Score composto: **30% frequência + 20% atraso + 35% ML + 15% misto**, "
            "com ajuste leve aos padrões estruturais."
        )
        if st.button("Correr ensemble", key="alg_ensemble"):
            with st.spinner("A calcular ensemble..."):
                res = analyzer.gerar_ensemble(
                    sorteio=sorteio, n_opcoes=n_ops, seed=int(seed)
                )
            for i, op in enumerate(res["opcoes"], 1):
                st.markdown(f"**Opção {i}**")
                st.markdown(bolas_html(op), unsafe_allow_html=True)
            st.markdown("**Top 20 do ranking composto**")
            st.dataframe(
                pd.DataFrame(
                    [{"Número": n, "Score": round(s, 4)} for n, s in res["ranking_top20"]]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Pesos: {res['componentes']}")

    with tab3:
        st.markdown(
            "K-means sobre a presença dos números em cada sorteio — "
            "**curiosidade exploratória**, não previsão."
        )
        k = st.slider("Nº de clusters", 2, 8, 4, key="alg_k")
        if st.button("Correr clustering", key="alg_cluster"):
            with st.spinner("K-means..."):
                cl = analyzer.clustering_sorteios(n_clusters=k, sorteio=sorteio)
            if not cl.get("ok"):
                st.warning(cl.get("mensagem", "Falhou."))
            else:
                st.success(cl["mensagem"])
                st.caption(f"{cl['n_sorteios']} sorteios · {cl['n_clusters']} clusters")
                for c in cl["clusters"]:
                    nums = ", ".join(
                        f"{n} ({p:.2f})" for n, p in c["numeros_caracteristicos"][:8]
                    )
                    st.markdown(
                        f"**Cluster {c['id']}** — {c['tamanho']} sorteios "
                        f"({c['pct']:.1f}%)\n\nNúmeros característicos: {nums or '—'}"
                    )

    with tab4:
        st.markdown(
            "Para cada sorteio recente: gera 1 combinação **só com dados anteriores** "
            "e conta quantos números acertaram. Compara com **aleatório**."
        )
        n_bt = st.slider("Últimos N sorteios", 5, 50, 20, key="alg_bt_n")
        if st.button("Correr backtest", key="alg_bt"):
            with st.spinner("Backtest rolante (pode demorar)..."):
                resumo = analyzer.resumo_backtest_algoritmos(
                    n_ultimos=n_bt, sorteio=sorteio, seed=int(seed)
                )
            if not resumo.get("ok"):
                st.warning(resumo.get("mensagem", "Sem dados."))
            else:
                st.markdown(resumo.get("mensagem", ""))
                medias = resumo["medias"]
                df_m = pd.DataFrame(
                    [
                        {
                            "Algoritmo": alg,
                            "Média acertos": round(info["media"], 3),
                            "Desvio": round(info["desvio"], 3),
                            "N": info["n"],
                        }
                        for alg, info in medias.items()
                    ]
                ).sort_values("Média acertos", ascending=False)
                st.dataframe(df_m, hide_index=True, use_container_width=True)
                st.bar_chart(df_m.set_index("Algoritmo")["Média acertos"])
                base = medias.get("aleatorio", {}).get("media")
                if base is not None:
                    melhores = [
                        (a, i["media"]) for a, i in medias.items() if a != "aleatorio"
                    ]
                    if melhores and max(m for _, m in melhores) <= base + 0.15:
                        st.warning(
                            "Nenhum algoritmo superou claramente o baseline aleatório. "
                            "Isto é o esperado se os sorteios forem essencialmente aleatórios."
                        )
                    elif melhores:
                        top_a, top_m = max(melhores, key=lambda x: x[1])
                        st.info(
                            f"Melhor no recorte: **{top_a}** ({top_m:.2f}/5). "
                            f"Aleatório: {base:.2f}/5. Diferenças pequenas costumam ser ruído."
                        )
                with st.expander("Detalhe por sorteio"):
                    st.dataframe(resumo["detalhe"], use_container_width=True, height=320)

        st.caption(
            "Backtest honesto: treino só com passado. Superar o aleatório de forma "
            "consistente em lotaria 5/90 é extremamente difícil — e raro."
        )


def pagina_estatisticas(analyzer: AnalisadorLotaria) -> None:

    """Mostro frequências, quentes/frios, somas e paridade."""
    st.subheader("Estatísticas e análise")
    st.caption(
        "Frequência absoluta, números quentes/frios por período, "
        "distribuição de somas e paridade par/ímpar."
    )
    if analyzer.df.empty:
        st.info("Sem dados. Adiciona resultados na barra lateral.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        sorteio_f = st.selectbox(
            "Sorteio",
            ["Todos"] + SORTEIOS,
            key="est_sorteio",
        )
    with c2:
        periodo = st.selectbox(
            "Período",
            [
                "Todo o histórico",
                "Últimos 30 dias",
                "Últimos 3 meses",
                "Últimos 6 meses",
                "Último ano",
            ],
            key="est_periodo",
        )
    with c3:
        top_n = st.slider("Top N (quentes/frios)", 5, 30, 15, key="est_top")

    sorteio = None if sorteio_f == "Todos" else sorteio_f
    meses_map = {
        "Todo o histórico": None,
        "Últimos 30 dias": None,  # tratado à parte
        "Últimos 3 meses": 3,
        "Últimos 6 meses": 6,
        "Último ano": 12,
    }
    meses = meses_map[periodo]

    # Nos últimos 30 dias filtro por data explicitamente
    data_inicio = None
    if periodo == "Últimos 30 dias":
        fim = analyzer.df["Data"].max()
        data_inicio = (fim - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    # ---- 1. Frequência absoluta ----
    st.markdown("### 1. Frequência absoluta")
    if data_inicio:
        freq = analyzer.frequencias_periodo(sorteio, data_inicio=data_inicio)
        df_p = analyzer._df_periodo(sorteio, data_inicio=data_inicio)
    else:
        freq = analyzer.frequencias_periodo(sorteio, meses=meses)
        df_p = analyzer._df_periodo(sorteio, meses=meses)

    st.caption(f"{len(df_p)} sorteios no recorte · {int(freq.sum())} aparições de números")
    df_freq = pd.DataFrame({
        "Número": freq.index.astype(int),
        "Vezes": freq.values.astype(int),
    }).sort_values("Número")
    st.bar_chart(df_freq.set_index("Número")["Vezes"], height=280)
    with st.expander("Tabela completa 1–90"):
        st.dataframe(
            df_freq.sort_values("Vezes", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    # ---- 2. Quentes / frios ----
    st.markdown("### 2. Números quentes e frios")
    st.caption("Comparação: período seleccionado vs. histórico total.")
    qf = analyzer.quentes_frios(sorteio=sorteio, meses=meses if not data_inicio else None, top=top_n)
    if data_inicio:
        # Recalculo quentes/frios no filtro de 30 dias
        freq_p = freq
        freq_t = analyzer.frequencias(sorteio)
        qf["quentes_periodo"] = [
            (int(n), int(v)) for n, v in freq_p.sort_values(ascending=False).head(top_n).items()
        ]
        qf["frios_periodo"] = [
            (int(n), int(v)) for n, v in freq_p.sort_values(ascending=True).head(top_n).items()
        ]
        qf["n_sorteios_periodo"] = len(df_p)

    col_q, col_f = st.columns(2)
    with col_q:
        st.markdown("**Quentes (mais frequentes)**")
        st.markdown("*Período*")
        for n, v in qf["quentes_periodo"]:
            st.caption(f"{n} → **{v}**x")
        st.markdown("*Total*")
        for n, v in qf["quentes_total"][:top_n]:
            st.caption(f"{n} → **{v}**x")
    with col_f:
        st.markdown("**Frios (menos frequentes)**")
        st.markdown("*Período*")
        for n, v in qf["frios_periodo"]:
            st.caption(f"{n} → **{v}**x")
        st.markdown("*Total*")
        for n, v in qf["frios_total"][:top_n]:
            st.caption(f"{n} → **{v}**x")

    # ---- 3. Distribuição de somas ----
    st.markdown("### 3. Distribuição de somas")
    st.caption(
        "Soma dos 5 números de cada sorteio. Em jogos 5/90 a distribuição "
        "aproxima-se de uma curva em sino — combinações só com números muito "
        "baixos ou muito altos são historicamente raras."
    )
    if data_inicio:
        dist = analyzer.distribuicao_somas(sorteio)
        # Filtro as somas do período
        somas_p = df_p[COLUNAS_NUMEROS].sum(axis=1).astype(int) if not df_p.empty else pd.Series(dtype=int)
        if len(somas_p):
            dist = {
                "somas": somas_p.tolist(),
                "media": float(somas_p.mean()),
                "desvio": float(somas_p.std(ddof=0)) if len(somas_p) > 1 else 0.0,
                "min": int(somas_p.min()),
                "max": int(somas_p.max()),
                "n": len(somas_p),
                "percentis": {
                    "p5": float(somas_p.quantile(0.05)),
                    "p25": float(somas_p.quantile(0.25)),
                    "p50": float(somas_p.quantile(0.50)),
                    "p75": float(somas_p.quantile(0.75)),
                    "p95": float(somas_p.quantile(0.95)),
                },
            }
    else:
        dist = analyzer.distribuicao_somas(sorteio, meses=meses)

    if dist["n"] == 0:
        st.warning("Sem dados neste recorte.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sorteios", dist["n"])
        m2.metric("Média da soma", f"{dist['media']:.1f}")
        m3.metric("Desvio (σ)", f"{dist['desvio']:.1f}")
        m4.metric("Min – Máx", f"{dist['min']} – {dist['max']}")
        # Montei um histograma simples com faixas
        somas_s = pd.Series(dist["somas"])
        bins = list(range(
            (dist["min"] // 10) * 10,
            ((dist["max"] // 10) + 2) * 10,
            10,
        ))
        if len(bins) < 2:
            bins = [dist["min"], dist["max"] + 1]
        cats = pd.cut(somas_s, bins=bins, right=False)
        hist = cats.value_counts().sort_index()
        hist_df = pd.DataFrame({
            "Faixa": [str(i) for i in hist.index],
            "Contagem": hist.values,
        })
        st.bar_chart(hist_df.set_index("Faixa")["Contagem"], height=260)
        p = dist["percentis"]
        st.caption(
            f"Percentis: P5={p['p5']:.0f} · P25={p['p25']:.0f} · "
            f"P50={p['p50']:.0f} · P75={p['p75']:.0f} · P95={p['p95']:.0f}"
        )

        st.markdown("**Avaliar uma combinação (soma)**")
        cols_n = st.columns(5)
        nums_teste = []
        defaults = [5, 12, 30, 47, 81]
        for i, col in enumerate(cols_n):
            with col:
                nums_teste.append(
                    st.number_input(
                        f"N{i+1}",
                        min_value=NUMERO_MINIMO,
                        max_value=NUMERO_MAXIMO,
                        value=defaults[i],
                        key=f"est_soma_n{i}",
                    )
                )
        if len(set(nums_teste)) == 5:
            av = analyzer.avaliar_soma_combinacao(list(nums_teste))
            st.info(av["mensagem"])
        else:
            st.caption("Escolhe 5 números distintos para avaliar a soma.")

    # ---- 4. Paridade ----
    st.markdown("### 4. Paridade (par / ímpar)")
    if data_inicio:
        par = analyzer.distribuicao_paridade(sorteio)
        # Recalculo a paridade no recorte df_p
        if not df_p.empty:
            padroes = {}
            tp = ti = 0
            for _, row in df_p.iterrows():
                nums = [int(row[c]) for c in COLUNAS_NUMEROS]
                np_ = sum(1 for n in nums if n % 2 == 0)
                ni = 5 - np_
                tp += np_
                ti += ni
                ch = f"{np_} pares + {ni} ímpares"
                padroes[ch] = padroes.get(ch, 0) + 1
            par = {
                "n": len(df_p),
                "padroes": dict(sorted(padroes.items(), key=lambda x: -x[1])),
                "total_pares": tp,
                "total_impares": ti,
                "media_pares_por_sorteio": tp / len(df_p),
                "pct_pares": 100.0 * tp / (tp + ti) if (tp + ti) else 0,
            }
    else:
        par = analyzer.distribuicao_paridade(sorteio, meses=meses)

    if par["n"] == 0:
        st.warning("Sem dados.")
    else:
        a, b, c = st.columns(3)
        a.metric("Sorteios", par["n"])
        b.metric("% números pares", f"{par['pct_pares']:.1f}%")
        c.metric("Média de pares / sorteio", f"{par['media_pares_por_sorteio']:.2f}")
        st.markdown("**Padrões mais comuns (quantos pares + ímpares por sorteio)**")
        pad_df = pd.DataFrame(
            [{"Padrão": k, "Vezes": v, "%": round(100 * v / par["n"], 1)} for k, v in par["padroes"].items()]
        )
        st.bar_chart(pad_df.set_index("Padrão")["Vezes"], height=240)
        st.dataframe(pad_df, use_container_width=True, hide_index=True)

    st.caption(
        "Estas estatísticas descrevem o histórico — não prevêem o próximo sorteio. "
        "Cada extracção é independente."
    )


def pagina_sobre() -> None:


    st.subheader("Sobre este projecto")
    st.markdown(
        """
**Loto** é uma ferramenta de
análise histórica dos sorteios da Lotaria Nacional de Angola
(Fezada, Aqueceu, Kazola, Eskebra).

**O que calcula**
- Frequência histórica de cada número (1-90)
- Atraso (sorteios desde a última aparição)
- Pares e trios que mais saem juntos
- Ranking com frequência + atraso + Random Forest
- Backtest das sugestões no passado
- Análise do teu bilhete face ao histórico

**Arquitectura**
- Lógica de negócio em `lotaria_core.py` (partilhada com a interface Tkinter)
- Persistência: Supabase (produção) ou CSV/JSON local (offline)\n- Sugestões bloqueadas e histórico de acertos também no Supabase (tabela `sugestoes_bloqueadas`)
- Autenticação simples via senha nos secrets
- Testes com pytest
        """
    )


# ---------------------------------------------------------------------------
# Assistente — personalidade, base de conhecimento e memória da conversa
# ---------------------------------------------------------------------------

ASSISTENTE_NOME = "Lota"
ASSISTENTE_PERSONA = (
    "Sou a Lota, assistente do app **Loto**. "
    "Ajudo com os dados dos sorteios (Fezada, Aqueceu, Kazola e Eskebra) "
    "e a navegar no app. Falo português de Angola, de forma directa e honesta."
)

DISCLAIMER_SUGESTOES = (
    "\n\n_⚠️ Isto é análise estatística do histórico — **não é previsão garantida**. "
    "Sorteios são aleatórios. Joga com responsabilidade._"
)

MENU_AJUDA = """**Menu · o que posso responder**

1. **Sobre o app** — *o que é o Loto?* · *como funciona?* · *para que serve?*
2. **Secções** — *o que faz o Dashboard?* · *como usar a Grelha?*
3. **Horários** — *a que horas é a Fezada?* · *quando saem as sugestões?*
4. **Sugestões** — *sugestões de hoje* · *opções da Kazola*
5. **Número específico** — *o 45 já saiu muito?* · *quando saiu o 23?*
6. **Quentes / frios** — *números mais frequentes* · *mais atrasados*
7. **Pares e trios** — *que números costumam sair juntos?*
8. **Resultados** — *últimos resultados* · *o que saiu no dia 10?*
9. **Bilhete** — *os meus números 5, 12, 30, 47, 81 já saíram juntos?*
10. **Fiabilidade** — *isto é fiável?* · *quantos acertos teve o modelo?*
11. **Método** — *como calculas as sugestões?* · *é Machine Learning?*
12. **Comparar sorteios** — *qual sorteio tem o número mais atrasado?*
13. **Adicionar dados** — *como meter um resultado?*
14. **Jogo responsável** — *isto vicia?* · *é seguro apostar assim?*

Escreve em português natural (mesmo com calão ou gralhas) — eu tento perceber."""

CONHECIMENTO_APP = {
    "o_que_e": (
        "**Loto** é uma ferramenta de análise histórica dos sorteios da "
        "Lotaria Nacional de Angola (LOTO 5/90: Fezada, Aqueceu, Kazola, Eskebra).\n\n"
        "Não é o site oficial da lotaria e **não garante acertos**. "
        "Usa frequência, atraso e um modelo Random Forest para gerar "
        "**4 opções de 5 números** por sorteio — só com base no histórico que tu (ou a base) registas."
    ),
    "como_usar": (
        "**Como usar o app (barra lateral / navegação):**\n\n"
        "1. **Dashboard** — 4 sugestões do dia (liberadas 2h30min antes) e partilha\n"
        "2. **Análise por Sorteio** — frequência, atraso, pares/trios de um sorteio\n"
        "3. **Grelha 1-90** — mapa de quentes e frios\n"
        "4. **Estatísticas** — frequência absoluta, quentes/frios, somas e paridade\n"
        "5. **Histórico** — resultados vencedores, qualidade dos dados, importar CSV\n"
        "5. **Histórico Sugestões** — o que foi sugerido e quantos acertos teve\n"
        "6. **Verificar números** — analisa o teu bilhete face ao histórico\n"
        "7. **Modelo ML** — retreino, features e backtest\n"
        "8. **Assistente** — sou eu; pergunta em português natural\n"
        "9. **Relatório Semanal** — resumo dos últimos 7 dias\n\n"
        "Na barra lateral: **adicionar resultados** (com senha, se configurada) e **modo escuro**."
    ),
    "horarios": (
        "Tudo em **hora de Angola (WAT)**:\n\n"
        "| Sorteio | Hora do sorteio | Sugestões a partir de |\n"
        "|---------|----------------|------------------------|\n"
        "| Fezada  | 10:00          | 07:30                  |\n"
        "| Aqueceu | 13:00          | 10:30                  |\n"
        "| Kazola  | 16:00          | 13:30                  |\n"
        "| Eskebra | 19:00          | 16:30                  |\n"
    ),
    "metodo": (
        "**Como calculo as sugestões**\n\n"
        "1. **Estatística clássica**: frequência total, frequência recente, atraso "
        "(sorteios sem aparecer) e se o número é par/ímpar.\n"
        "2. **Machine Learning a sério**: um **Random Forest Regressor** treinado "
        "no histórico — aprende padrões de combinação dessas features.\n"
        "3. **Mistura**: cerca de **65% ML + 35% estatística** no ranking final.\n"
        "4. As 4 opções são fatias do ranking (1.ª = top 5, 2.ª = 6.º–10.º, etc.).\n\n"
        "Depois de mostradas, as sugestões de um dia **congelam** — não mudam quando "
        "lanças o resultado. Os dados novos só melhoram os sorteios seguintes.\n\n"
        "Isto **não prevê o futuro**: só resume o passado. Cada sorteio é independente."
    ),
    "jogo_responsavel": (
        "**Jogo responsável**\n\n"
        "A lotaria é entretenimento, não forma de ganhar a vida. "
        "Nenhuma análise estatística (incluindo a deste app) altera a natureza aleatória dos sorteios.\n\n"
        "- Define um limite de gasto e **respeita-o**.\n"
        "- Não persigas perdas nem aumentes apostas para “recuperar”.\n"
        "- Se sentes que o jogo está a controlar o teu tempo, dinheiro ou humor, "
        "pausa e fala com alguém de confiança.\n"
        "- Em Angola, podes procurar apoio junto de serviços de saúde ou linhas de apoio locais.\n\n"
        "O **Loto** existe para curiosidade e organização de dados — não para incentivar apostas excessivas."
    ),
    "adicionar": (
        "Para **adicionar um resultado** usa a barra lateral:\n\n"
        "1. Escolhe a data e o sorteio\n"
        "2. Preenche os 5 números (ou fotografa o boletim — o OCR tenta ler)\n"
        "3. Clica em **Salvar Resultado**\n\n"
        "Se já existir registo nessa data/sorteio com números **diferentes**, "
        "o app avisa e pede confirmação antes de substituir.\n"
        "Se pedir senha, é a `APP_PASSWORD` nos secrets."
    ),
    "limites": (
        "Sou a assistente deste app de análise.\n\n"
        "Não represento a Lotaria Nacional, não pago prémios e não dou garantias de acerto. "
        "Para regras e resultados oficiais, usa os canais da Lotaria Nacional."
    ),
}


def _extrair_numeros_mensagem(texto: str) -> list[int]:
    """Extraí números 1–90 da frase (evito anos tipo 2026)."""
    encontrados = []
    for m in re.findall(r"\b(\d{1,2})\b", texto):
        n = int(m)
        if NUMERO_MINIMO <= n <= NUMERO_MAXIMO:
            encontrados.append(n)
    # únicos preservando ordem
    vistos = set()
    out = []
    for n in encontrados:
        if n not in vistos:
            vistos.add(n)
            out.append(n)
    return out


def _extrair_data_mensagem(texto: str) -> str | None:
    """Tentei achar uma data na pergunta (YYYY-MM-DD, DD/MM/YYYY, DD/MM, 'dia 10')."""
    t = texto.strip()
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{y}-{mo:02d}-{d:02d}"
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            y = hoje_angola()[:4]
            return f"{y}-{mo:02d}-{d:02d}"
    m = re.search(r"\bdia\s+(\d{1,2})\b", t, re.I)
    if m:
        d = int(m.group(1))
        if 1 <= d <= 31:
            hoje = hoje_angola()
            y, mo = hoje[:4], hoje[5:7]
            return f"{y}-{mo}-{d:02d}"
    return None


def _detectar_sorteio(texto: str):
    texto_l = texto.lower()
    for s in SORTEIOS:
        if s.lower() in texto_l:
            return s
    return None


def _intent_scores(texto: str) -> dict[str, float]:
    """Pontuei os intents (em vez de o primeiro match ganhar)."""
    p = texto.lower().strip()
    scores: dict[str, float] = {}

    def add(intent: str, pts: float):
        scores[intent] = scores.get(intent, 0.0) + pts

    # menu / ajuda (prioridade alta)
    if any(k in p for k in ("ajuda", "help", "menu", "comandos", "o que podes", "o que pode", "o que você pode")):
        add("menu", 5.0)

    if any(k in p for k in ("o que é", "o que e", "que app", "para que serve", "o que faz este", "o que faz o app")):
        add("o_que_e", 4.0)
    if any(k in p for k in ("como usar", "como funciona", "como se usa", "tutorial")):
        add("como_usar", 4.0)
    if any(k in p for k in ("hora", "horário", "horario", "quando abre", "a que horas", "liberação", "liberacao")):
        add("horarios", 3.5)
    if any(k in p for k in ("como calcul", "como fazes", "método", "metodo", "random forest", "machine learning", "é ml", "e ml", "ml a sério", "ml a serio")):
        add("metodo", 4.5)
    if any(k in p for k in (
        "fiáv", "fiav", "fiável", "fiavel", "precisão", "precisao", "acertos já", "acertos do modelo",
        "backtest", "confiáv", "confiav", "funciona mesmo", "ganho se", "quantos acertos",
    )):
        add("fiabilidade", 5.0)
    if any(k in p for k in (
        "vicia", "vício", "vicio", "gastar muito", "jogo responsável", "jogo responsavel",
        "é seguro apostar", "e seguro apostar", "apostar assim", "perder dinheiro", "addiction",
    )):
        add("jogo_responsavel", 5.5)
    if any(k in p for k in ("adicionar", "lançar", "lancar", "gravar resultado", "salvar resultado", "meter resultado", "ocr", "foto do boletim")):
        add("adicionar", 3.5)
    if any(k in p for k in ("oficial", "funcionário", "funcionario", "garantia", "promessa", "vou ganhar")):
        add("limites", 3.0)

    if any(k in p for k in (
        "sugest", "palpite", "jogar hoje", "números de hoje", "numeros de hoje",
        "4 opções", "4 opcoes", "três opções", "ranking para",
    )):
        add("sugestoes", 4.0)
    # "opção/opções" sozinho é fraco (colide com navegação)
    if re.search(r"\bop[cç][aã]o(es)?\b", p) and "histórico" not in p and "historico" not in p:
        add("sugestoes", 2.0)

    if any(k in p for k in ("mais frequente", "mais sai", "quente", "mais sorteado", "top número", "top numero")):
        add("frequencia", 4.0)
    if any(k in p for k in ("atrasado", "atraso", "frio", "não sai", "nao sai", "esquecido", "há quanto tempo")):
        add("atraso", 4.0)
    if any(k in p for k in ("juntos", "par ", "pares", "trio", "trios", "costumam sair", "saem juntos", "combina")):
        add("pares", 4.5)

    if any(k in p for k in ("meu bilhete", "meus números", "meus numeros", "verificar bilhete", "já saíram juntos", "ja sairam juntos", "saíram juntos", "analisar bilhete")):
        add("bilhete", 5.0)

    if any(k in p for k in ("último", "ultimo", "últimos", "ultimos", "recente", "ontem")):
        add("ultimos", 3.0)
    if re.search(r"\b(dia\s+\d{1,2}|\d{1,2}[/-]\d{1,2}|20\d{2}-\d{2}-\d{2})\b", p) or "o que saiu" in p or "resultado do dia" in p or "o que deram" in p:
        add("data_especifica", 6.0)

    if any(k in p for k in ("qual sorteio", "comparar sorteio", "mais atrasado entre", "entre os sorteios")):
        add("comparar_sorteios", 4.0)

    if any(k in p for k in ("quantos sorteios", "tamanho da base", "quantos dados", "total na base")):
        add("contagem", 3.5)

    if any(k in p for k in ("grelha", "dashboard", "histórico", "historico", "relatório", "relatorio", "onde vejo", "onde fica", "secção", "seccao", "aba ")):
        add("navegacao", 3.0)

    if any(k in p for k in ("obrigado", "obrigada", "valeu", "thanks", "brigado")):
        add("agradecimento", 6.0)
    if any(k in p for k in ("olá", "ola", "oi", "bom dia", "boa tarde", "boa noite", "hey")):
        add("saudacao", 5.0)

    # número específico: há 1–2 números e palavras de frequência/atraso/última vez
    nums = _extrair_numeros_mensagem(texto)
    if "o que saiu" in p or "resultado do dia" in p or re.search(r"\bdia\s+\d{1,2}\b", p):
        pass  # deixa data_especifica ganhar
    elif len(nums) == 1 or (len(nums) <= 2 and any(k in p for k in ("saiu", "número", "numero", "o ", "e o "))):
        if any(k in p for k in ("saiu", "frequ", "atras", "última", "ultima", "quando", "quanto", "muito", "vezes")) or len(p) < 25:
            add("numero_especifico", 4.0 + (1.0 if len(nums) == 1 else 0))

    # bilhete se 5 números na frase
    if len(nums) >= 5:
        add("bilhete", 4.5)

    return scores


def _intent(texto: str) -> str:
    scores = _intent_scores(texto)
    if not scores:
        nums = _extrair_numeros_mensagem(texto)
        if len(nums) == 1:
            return "numero_especifico"
        if len(nums) >= 5:
            return "bilhete"
        return "desconhecido"
    return max(scores.items(), key=lambda x: x[1])[0]


def _memoria_curta() -> dict:
    msgs = st.session_state.get("chat_mensagens", [])
    return {
        "msgs": msgs,
        "ultimo_sorteio": st.session_state.get("chat_ultimo_sorteio"),
        "topicos": st.session_state.get("chat_topicos", {}),
        "nome_user": st.session_state.get("chat_nome_user"),
    }


def _aprender(intent: str, sorteio: str | None, pergunta: str = "") -> None:
    topicos = st.session_state.setdefault("chat_topicos", {})
    topicos[intent] = topicos.get(intent, 0) + 1
    if sorteio:
        st.session_state.chat_ultimo_sorteio = sorteio
    # Capto o nome do utilizador se ele se apresentar
    m = re.search(
        r"(?:chamo-?me|meu nome é|meu nome e|sou a?o?)\s+([A-Za-zÀ-ÿ]{2,20})",
        pergunta,
        re.I,
    )
    if m:
        st.session_state.chat_nome_user = m.group(1).strip().title()


def _tratar(nome: str | None, texto: str) -> str:
    if nome:
        return f"{nome}, {texto[0].lower() + texto[1:]}" if texto else texto
    return texto


def _info_numero(analyzer: AnalisadorLotaria, numero: int, sorteio: str | None) -> str:
    if analyzer.df.empty:
        return "Ainda não há histórico na base para falar desse número."
    freq = analyzer.frequencias(sorteio)
    atr = analyzer.atrasos(sorteio)
    f = int(freq.get(numero, 0))
    a = int(atr.get(numero, 0))
    df_local = analyzer.df if not sorteio else analyzer.df[analyzer.df["Sorteio"] == sorteio]
    ultima = None
    for _, row in df_local.sort_values("Data", ascending=False).iterrows():
        nums = {int(row[c]) for c in COLUNAS_NUMEROS}
        if numero in nums:
            d = row["Data"]
            ultima = d.strftime("%d/%m/%Y") if hasattr(d, "strftime") else str(d)[:10]
            break
    ambito = f" no sorteio **{sorteio}**" if sorteio else " (todos os sorteios)"
    linhas = [
        f"**Número {numero}**{ambito}:",
        f"- Saiu **{f}** vez(es) no histórico carregado",
        f"- Atraso actual: **{a}** sorteio(s) sem aparecer",
    ]
    if ultima:
        linhas.append(f"- Última vez: **{ultima}**")
    else:
        linhas.append("- Ainda não aparece neste recorte do histórico")
    total = len(df_local)
    if total:
        linhas.append(f"- Em {total} sorteios analisados → cerca de **{100 * f / total:.1f}%** das vezes")
    return "\n".join(linhas)


def _oferta_proativa(intent: str, memoria: dict) -> str:
    if intent in ("frequencia", "atraso") and memoria.get("ultimo_sorteio"):
        return f"\n\nQueres o mesmo só para **{memoria['ultimo_sorteio']}**?"
    if intent == "ultimos":
        return "\n\nPosso também mostrar as **4 sugestões** de um sorteio — é só dizer o nome."
    return ""


def _responder_assistente(pergunta: str, analyzer: AnalisadorLotaria) -> str:
    p = pergunta.strip()
    intent = _intent(p)
    sorteio = _detectar_sorteio(p) or st.session_state.get("chat_ultimo_sorteio")
    memoria = _memoria_curta()
    nome = memoria.get("nome_user")
    _aprender(intent, _detectar_sorteio(p), p)
    nums_msg = _extrair_numeros_mensagem(p)

    if intent == "menu":
        return MENU_AJUDA

    if intent == "saudacao":
        saud = f"Olá, {nome}! " if nome else "Olá! "
        return (
            f"{saud}Sou a **{ASSISTENTE_NOME}**, assistente do app **Loto**.\n\n"
            "Posso falar de sugestões, frequência, atraso, o teu bilhete, "
            "fiabilidade do modelo e horários. Escreve **ajuda** para ver o menu completo."
        )

    if intent == "agradecimento":
        return _tratar(nome, "De nada! Se precisares de mais alguma coisa, é só dizer — ou escreve **ajuda**.")

    if intent == "o_que_e":
        return CONHECIMENTO_APP["o_que_e"] + _oferta_proativa(intent, memoria)
    if intent == "como_usar":
        return CONHECIMENTO_APP["como_usar"]
    if intent == "horarios":
        return CONHECIMENTO_APP["horarios"]
    if intent == "metodo":
        return CONHECIMENTO_APP["metodo"]
    if intent == "jogo_responsavel":
        return CONHECIMENTO_APP["jogo_responsavel"]
    if intent == "adicionar":
        return CONHECIMENTO_APP["adicionar"]
    if intent == "limites":
        return CONHECIMENTO_APP["limites"]

    if intent == "navegacao":
        return CONHECIMENTO_APP["como_usar"]

    if intent == "contagem":
        n = len(analyzer.df)
        return _tratar(nome, f"Temos **{n}** sorteios registados na base neste momento.")

    if intent == "numero_especifico":
        if not nums_msg:
            return "Diz-me qual o número (1–90). Ex.: *o 45 já saiu muito?*"
        partes = [_info_numero(analyzer, n, _detectar_sorteio(p)) for n in nums_msg[:3]]
        return "\n\n".join(partes)

    if intent == "frequencia":
        if analyzer.df.empty:
            return "Ainda não há dados. Adiciona resultados na barra lateral."
        freq = analyzer.frequencias(_detectar_sorteio(p))
        top = freq.sort_values(ascending=False).head(10)
        rotulo = _detectar_sorteio(p) or "geral"
        linhas = [f"**{int(num)}** — {int(v)}x" for num, v in top.items()]
        return (
            f"Números mais frequentes ({rotulo}):\n\n- "
            + "\n- ".join(linhas)
            + _oferta_proativa(intent, memoria)
        )

    if intent == "atraso":
        if analyzer.df.empty:
            return "Ainda não há dados na base."
        atr = analyzer.atrasos(_detectar_sorteio(p))
        top = atr.sort_values(ascending=False).head(10)
        rotulo = _detectar_sorteio(p) or "geral"
        linhas = [f"**{int(num)}** — {int(v)} sorteios sem sair" for num, v in top.items()]
        return (
            f"Números mais atrasados ({rotulo}):\n\n- "
            + "\n- ".join(linhas)
            + _oferta_proativa(intent, memoria)
        )

    if intent == "pares":
        if analyzer.df.empty:
            return "Sem histórico ainda — não consigo calcular pares/trios."
        s = _detectar_sorteio(p)
        pares = analyzer.pares_frequentes(sorteio=s, tamanho=2, top=8)
        trios = analyzer.pares_frequentes(sorteio=s, tamanho=3, top=5)
        rotulo = s or "todos os sorteios"
        txt = [f"**Pares mais frequentes** ({rotulo}):"]
        if pares:
            txt += [f"- {a} e {b}: **{c}** vez(es)" for (a, b), c in pares]
        else:
            txt.append("- (sem dados)")
        txt.append(f"\n**Trios mais frequentes** ({rotulo}):")
        if trios:
            txt += [f"- {', '.join(map(str, t))}: **{c}** vez(es)" for t, c in trios]
        else:
            txt.append("- (sem dados)")
        return "\n".join(txt)

    if intent == "bilhete":
        if len(nums_msg) < 5:
            return (
                "Para verificar um bilhete preciso de **5 números** (1–90). "
                "Ex.: *os meus números 5, 12, 30, 47, 81 já saíram juntos?*"
            )
        try:
            info = analyzer.analisar_bilhete(nums_msg[:5], _detectar_sorteio(p))
        except ResultadoInvalidoError as err:
            return str(err)
        linhas = [
            f"Bilhete **{', '.join(map(str, info['numeros']))}**:",
            "",
        ]
        for d in info["detalhe"]:
            linhas.append(
                f"- {d['numero']}: frequência {d['frequencia']}, atraso {d['atraso']}"
            )
        if info["vezes_juntos"] > 0:
            linhas.append(
                f"\nSaíram **todos juntos** {info['vezes_juntos']} vez(es). "
                f"Última: **{info['ultima_vez_juntos']}**."
            )
        else:
            linhas.append("\nEstes 5 **nunca** saíram todos juntos neste histórico.")
        return "\n".join(linhas)

    if intent == "data_especifica":
        data = _extrair_data_mensagem(p)
        if not data:
            return "Indica a data (ex.: *o que saiu no dia 10?* ou *10/08/2026*)."
        if analyzer.df.empty:
            return "A base ainda está vazia."
        df_dia = analyzer.df[analyzer.df["Data"].dt.strftime("%Y-%m-%d") == data]
        if df_dia.empty:
            return f"Não tenho resultados registados para **{data}**."
        linhas = [f"**Resultados de {data}:**"]
        for _, row in df_dia.sort_values("Sorteio").iterrows():
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            hs, ms = HORA_SORTEIO.get(row["Sorteio"], (0, 0))
            linhas.append(
                f"- **{row['Sorteio']}** ({hs:02d}:{ms:02d}): {', '.join(map(str, nums))}"
            )
        return "\n".join(linhas)

    if intent == "ultimos":
        if analyzer.df.empty:
            return "Ainda não há resultados."
        ultimos = analyzer.df.sort_values("Data", ascending=False).head(8)
        linhas = []
        for _, row in ultimos.iterrows():
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            data = row["Data"].strftime("%d/%m/%Y") if hasattr(row["Data"], "strftime") else str(row["Data"])[:10]
            linhas.append(f"**{data}** · {row['Sorteio']}: {', '.join(map(str, nums))}")
        return "Últimos resultados:\n\n- " + "\n- ".join(linhas) + _oferta_proativa(intent, memoria)

    if intent == "comparar_sorteios":
        if analyzer.df.empty:
            return "Sem dados para comparar sorteios."
        linhas = ["**Comparação de atraso máximo por sorteio:**"]
        for s in SORTEIOS:
            atr = analyzer.atrasos(s)
            if atr.empty or atr.max() == 0:
                linhas.append(f"- **{s}**: sem dados suficientes")
                continue
            num = int(atr.idxmax())
            linhas.append(f"- **{s}**: número **{num}** com atraso **{int(atr.max())}**")
        return "\n".join(linhas)

    if intent == "fiabilidade":
        if analyzer.df.empty or len(analyzer.df) < 10:
            return (
                "Ainda há pouco histórico para falar de fiabilidade com seriedade. "
                "Quanto mais resultados registares, mais o backtest e as médias de acertos fazem sentido.\n\n"
                "Lembra-te: mesmo com muitos dados, **sorteios são aleatórios** — "
                "o modelo descreve o passado, não garante o futuro."
            )
        partes = ["**Fiabilidade (com honestidade):**\n"]
        try:
            stats = analyzer.estatisticas_acertos()
            if stats["n_sorteios"]:
                partes.append(
                    f"- Histórico de sugestões vs resultado real: "
                    f"**{stats['n_sorteios']}** sorteios avaliados"
                )
                if stats["media_melhor"] is not None:
                    partes.append(
                        f"- Média da **melhor** das 4 opções: "
                        f"**{stats['media_melhor']:.2f}** números certos (em 5)"
                    )
                if stats["media_todas_opcoes"] is not None:
                    partes.append(
                        f"- Média de **todas** as opções: "
                        f"**{stats['media_todas_opcoes']:.2f}**"
                    )
            else:
                partes.append(
                    "- Ainda não há sugestões congeladas confrontadas com resultados. "
                    "Usa o Dashboard e depois lança os resultados para acumular acertos."
                )
        except Exception:
            partes.append("- Não consegui ler as estatísticas de acertos neste momento.")
        try:
            df_bt = analyzer.backtest_sugestoes(n_ultimos=15)
            if df_bt is not None and not df_bt.empty and "Acertos" in df_bt.columns:
                media_bt = float(df_bt["Acertos"].mean())
                partes.append(
                    f"- Backtest (últimos sorteios, 1.ª sugestão): média **{media_bt:.2f}**/5"
                )
        except Exception:
            pass
        partes.append(
            "\nEstes números **não** significam que vais acertar amanhã. "
            "São só o desempenho no passado. Joga com responsabilidade."
        )
        return "\n".join(partes)

    if intent == "sugestoes":
        if analyzer.df.empty:
            return (
                "Ainda não há dados para gerar sugestões. "
                "Adiciona alguns resultados na barra lateral e volta a perguntar."
            )
        tres = analyzer.prever_tres_opcoes_por_sorteio()
        s = _detectar_sorteio(p)
        if s:
            ops = tres[s]
            linhas = [f"**Opção {i}**: {', '.join(map(str, nums))}" for i, nums in enumerate(ops, 1)]
            corpo = f"4 sugestões para **{s}**:\n\n- " + "\n- ".join(linhas)
        else:
            partes = []
            for s2 in SORTEIOS:
                ops = tres[s2]
                ops_txt = " · ".join(
                    [f"Op{i}: {', '.join(map(str, n))}" for i, n in enumerate(ops, 1)]
                )
                partes.append(f"**{s2}**: {ops_txt}")
            corpo = "4 sugestões por sorteio:\n\n" + "\n\n".join(partes)
        return corpo + DISCLAIMER_SUGESTOES

    # Resposta de recurso quando não reconheço o pedido
    if len(nums_msg) == 1:
        return _info_numero(analyzer, nums_msg[0], _detectar_sorteio(p))
    if len(nums_msg) >= 5:
        intent = "bilhete"
        # Reutilizo a lógica do bilhete
        try:
            info = analyzer.analisar_bilhete(nums_msg[:5], _detectar_sorteio(p))
            linhas = [f"Bilhete **{', '.join(map(str, info['numeros']))}**:"]
            for d in info["detalhe"]:
                linhas.append(f"- {d['numero']}: freq {d['frequencia']}, atraso {d['atraso']}")
            if info["vezes_juntos"]:
                linhas.append(f"Juntos: {info['vezes_juntos']}x (última {info['ultima_vez_juntos']})")
            else:
                linhas.append("Nunca saíram todos juntos neste histórico.")
            return "\n".join(linhas)
        except ResultadoInvalidoError as err:
            return str(err)

    return (
        (_tratar(nome, "Não apanhei bem o pedido.") if nome else "Não apanhei bem o pedido.")
        + "\n\nEscreve **ajuda** para ver o menu completo do que posso responder.\n\n"
        "Exemplos: *o 45 já saiu muito?* · *sugestões da Fezada* · "
        "*que números saem juntos?* · *isto é fiável?*"
    )


def _mensagem_boas_vindas_assistente() -> dict:
    return {
        "role": "assistant",
        "content": (
            f"Olá! Sou a **{ASSISTENTE_NOME}**, assistente do app **Loto**.\n\n"
            "Escreve **ajuda** para ver o menu completo.\n\n"
            "Exemplos rápidos:\n"
            "- *o 45 já saiu muito?*\n"
            "- *sugestões da Fezada*\n"
            "- *que números costumam sair juntos?*\n"
            "- *isto é fiável?*\n"
            "- *os meus números 5, 12, 30, 47, 81 já saíram juntos?*\n\n"
            "Falo directo e com honestidade estatística — sorteios são aleatórios."
        ),
    }


def pagina_assistente(analyzer: AnalisadorLotaria) -> None:
    st.subheader(f"Assistente · {ASSISTENTE_NOME}")
    st.caption(ASSISTENTE_PERSONA)

    col_limpar, _ = st.columns([1, 4])
    with col_limpar:
        if st.button("Limpar conversa", key="btn_limpar_chat"):
            st.session_state.chat_mensagens = [_mensagem_boas_vindas_assistente()]
            st.session_state.chat_topicos = {}
            st.session_state.pop("chat_ultimo_sorteio", None)
            st.session_state.pop("chat_nome_user", None)
            st.rerun()

    if "chat_mensagens" not in st.session_state:
        st.session_state.chat_mensagens = [_mensagem_boas_vindas_assistente()]
    if "chat_topicos" not in st.session_state:
        st.session_state.chat_topicos = {}

    for msg in st.session_state.chat_mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escreve a tua pergunta..."):
        st.session_state.chat_mensagens.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        resposta = _responder_assistente(prompt, analyzer)
        st.session_state.chat_mensagens.append({"role": "assistant", "content": resposta})
        with st.chat_message("assistant"):
            st.markdown(resposta)


def pagina_relatorio_semanal(analyzer: AnalisadorLotaria) -> None:
    st.subheader("Relatório Semanal")
    st.caption(
        "Resumo automático dos últimos 7 dias com dados. "
        "Útil para acompanhar tendências de frequência e atraso."
    )

    df = analyzer.df.copy()
    if df.empty:
        st.warning("Ainda não há dados para gerar o relatório.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df["Data"]):
        df["Data"] = pd.to_datetime(df["Data"])

    hoje = pd.Timestamp.now().normalize()
    inicio_semana = hoje - pd.Timedelta(days=6)
    inicio_semana_ant = inicio_semana - pd.Timedelta(days=7)

    df_semana = df[df["Data"] >= inicio_semana]
    df_semana_ant = df[(df["Data"] >= inicio_semana_ant) & (df["Data"] < inicio_semana)]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sorteios nesta semana", len(df_semana))
    with c2:
        st.metric("Sorteios semana anterior", len(df_semana_ant))
    with c3:
        st.metric("Total na base", len(df))

    if df_semana.empty:
        st.info(
            f"Nenhum resultado entre {inicio_semana.strftime('%Y-%m-%d')} e "
            f"{hoje.strftime('%Y-%m-%d')}. Adiciona resultados recentes para ver o relatório."
        )
        return

    st.markdown(
        f"**Período:** {inicio_semana.strftime('%Y-%m-%d')} → {hoje.strftime('%Y-%m-%d')}"
    )

    st.markdown("### Números mais frequentes da semana")
    matriz = df_semana[COLUNAS_NUMEROS].to_numpy()
    contagem = pd.Series(0, index=range(NUMERO_MINIMO, NUMERO_MAXIMO + 1), dtype=int)
    for num in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
        contagem[num] = int((matriz == num).any(axis=1).sum())
    top_freq = contagem.nlargest(10)
    top_freq = top_freq[top_freq > 0]

    if top_freq.empty:
        st.write("Sem aparições suficientes.")
    else:
        cols = st.columns(min(5, len(top_freq)))
        for i, (num, cnt) in enumerate(top_freq.items()):
            with cols[i % len(cols)]:
                st.metric(f"Nº {int(num)}", f"{int(cnt)}x")

    st.markdown("### Resultados por sorteio (esta semana)")
    for sorteio in SORTEIOS:
        sub = df_semana[df_semana["Sorteio"] == sorteio].sort_values("Data", ascending=False)
        if sub.empty:
            st.markdown(f"**{sorteio}**: sem resultados esta semana")
            continue
        linhas = []
        for _, row in sub.iterrows():
            nums = [int(row[c]) for c in COLUNAS_NUMEROS]
            data = row["Data"].strftime("%Y-%m-%d")
            linhas.append(f"`{data}` → {', '.join(map(str, nums))}")
        st.markdown(f"**{sorteio}** ({len(sub)} sorteios)\n\n- " + "\n- ".join(linhas))

    if not df_semana_ant.empty:
        st.markdown("### Comparação com a semana anterior")
        matriz_ant = df_semana_ant[COLUNAS_NUMEROS].to_numpy()
        cont_ant = pd.Series(0, index=range(NUMERO_MINIMO, NUMERO_MAXIMO + 1), dtype=int)
        for num in range(NUMERO_MINIMO, NUMERO_MAXIMO + 1):
            cont_ant[num] = int((matriz_ant == num).any(axis=1).sum())

        diff = contagem - cont_ant
        subiram = diff.nlargest(5)
        desceram = diff.nsmallest(5)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mais “aquecidos” (subiram de frequência)**")
            for num, d in subiram.items():
                if d > 0:
                    st.write(f"Nº **{int(num)}**: {int(d):+d}")
        with c2:
            st.markdown("**Mais “esfriados” (desceram de frequência)**")
            for num, d in desceram.items():
                if d < 0:
                    st.write(f"Nº **{int(num)}**: {int(d):+d}")

    st.divider()
    resumo = df_semana[["Data", "Sorteio"] + COLUNAS_NUMEROS].copy()
    resumo["Data"] = resumo["Data"].dt.strftime("%Y-%m-%d")
    csv = resumo.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descarregar relatório da semana (CSV)",
        data=csv,
        file_name=f"relatorio_semanal_{inicio_semana.strftime('%Y%m%d')}_{hoje.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.caption(
        "Nota: este relatório é gerado sob demanda a partir dos dados já registados. "
        "Em Streamlit Community Cloud não há cron nativo; para envio automático por e-mail "
        "seria necessário um serviço externo (GitHub Actions, n8n, etc.)."
    )


PAGINAS = {
    "Dashboard": pagina_dashboard,
    "Análise por Sorteio": pagina_analise_por_sorteio,
    "Grelha 1-90": pagina_grelha,
    "Estatísticas": pagina_estatisticas,
    "Algoritmos": pagina_algoritmos,
    "Histórico": pagina_historico,
    "Histórico Sugestões": pagina_historico_sugestoes,
    "Verificar números": pagina_verificar_numeros,
    "Modelo ML": pagina_modelo_ml,
    "Assistente": pagina_assistente,
    "Relatório Semanal": pagina_relatorio_semanal,
}

# Estas páginas funcionam mesmo sem dados na base
PAGINAS_SEM_DADOS = {"Histórico", "Histórico Sugestões", "Assistente", "Verificar números", "Sobre"}


def cabecalho() -> None:
    st.markdown(
        """
        <div style="margin-bottom:0.5rem;">
            <h1 style="margin:0; color:#0a4d8c; font-size:1.6rem;">Loto</h1>
            <p style="margin:0; color:#555; font-size:0.85rem;">
                4 opções por sorteio · liberadas 2h30min antes
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _aplicar_tema_escuro() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117 !important; color: #e8e8e8 !important; }
        h1, h2, h3, p, label, span, .stMarkdown { color: #e8e8e8 !important; }
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
            color: #e8e8e8 !important;
        }
        .linha-sorteio .rotulo { color: #7eb6e8 !important; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a2a4a 0%, #061828 100%) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    analyzer = get_analyzer()

    tema_escuro = st.sidebar.toggle("Modo escuro", value=False, key="tema_escuro")
    if tema_escuro:
        _aplicar_tema_escuro()

    cabecalho()

    pagina = st.sidebar.radio("Navegação", list(PAGINAS.keys()) + ["Sobre"])
    st.sidebar.markdown("---")

    if verificar_senha():
        secao_adicionar_resultado(analyzer)
    else:
        st.sidebar.info("Faz login na barra lateral para adicionar resultados.")

    if pagina == "Sobre":
        pagina_sobre()
        return

    if analyzer.df.empty and pagina not in PAGINAS_SEM_DADOS:
        st.info(
            "Ainda não há resultados na base. "
            "**Adiciona o primeiro** na barra lateral, ou importa um CSV em **Histórico**."
        )
        return

    if not analyzer.df.empty:
        st.success(f"Total de sorteios na base: {len(analyzer.df)}")

    PAGINAS[pagina](analyzer)


_ = main()
