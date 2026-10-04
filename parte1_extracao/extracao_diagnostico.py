"""
CardioIA - Fase 2 - Parte 1: Extração de sintomas e sugestão de diagnóstico.

Sistema BASEADO EM REGRAS (sem machine learning):
  frase do paciente -> procura expressões do mapa -> soma pontos por doença -> sugestão

Como rodar (dentro da pasta parte1_extracao):
    python extracao_diagnostico.py
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).parent  # caminhos relativos à pasta do script
ARQ_FRASES = PASTA / "frases_sintomas.txt"
ARQ_MAPA = PASTA / "mapa_conhecimento.csv"
# casos_dificeis.txt é escrito pela PESSOA 3 (P3), sem ela ver o mapa de conhecimento:
# são ~5 frases difíceis (sintomas de duas doenças, negação, jeito popular de falar).
# Este arquivo ainda pode não existir; nesse caso a seção de casos difíceis é pulada.
ARQ_DIFICEIS = PASTA / "casos_dificeis.txt"
ARQ_SAIDA = PASTA / "resultados.csv"
ARQ_SAIDA_DIFICEIS = PASTA / "resultados_casos_dificeis.csv"

SEM_SINTOMA = "nenhum sintoma reconhecido, encaminhar para avaliação"


# ---------------------------------------------------------------- Etapa 1
def normalizar(texto: str) -> str:
    """Minúsculas + sem acento + pontuação vira espaço. 'Tórax' == 'torax'."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto.lower())
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------------- Etapa 2
def carregar_mapa(caminho=ARQ_MAPA) -> list[tuple[str, str]]:
    """Lê o CSV e devolve pares únicos (expressão normalizada, doença)."""
    df = pd.read_csv(caminho, encoding="utf-8")
    pares = set()
    for _, linha in df.iterrows():
        for col in ("sintoma_1", "sintoma_2"):
            if pd.notna(linha[col]) and str(linha[col]).strip():
                pares.add((normalizar(linha[col]), linha["doenca_associada"]))
    return sorted(pares)


def carregar_frases(caminho) -> list[str]:
    """Lê o .txt (uma frase por linha), ignorando linhas vazias."""
    with open(caminho, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ---------------------------------------------------------------- Etapa 3
def achar_sintomas(frase_norm: str, pares) -> dict:
    """
    Devolve {expressão: [doenças]} para cada expressão presente na frase.
    O \\b evita casar pedaço de palavra (ex.: 'ar' dentro de 'parar').
    """
    achados = {}
    for expr, doenca in pares:
        if re.search(rf"\b{re.escape(expr)}\b", frase_norm):
            achados.setdefault(expr, []).append(doenca)
    return achados


# ---------------------------------------------------------------- Etapa 4
def pontuar(achados: dict) -> list[tuple[str, int]]:
    """+1 por sintoma encontrado para cada doença ligada a ele. Ordem decrescente."""
    pontos = {}
    for doencas in achados.values():
        for d in doencas:
            pontos[d] = pontos.get(d, 0) + 1
    return sorted(pontos.items(), key=lambda x: (-x[1], x[0]))


def sugerir(ranking: list[tuple[str, int]]) -> tuple[str, int, str]:
    """Retorna (diagnóstico, pontuação, observação sobre a 2ª colocada)."""
    if not ranking:
        return SEM_SINTOMA, 0, ""
    diag, pts = ranking[0]
    obs = ""
    if len(ranking) > 1:
        d2, p2 = ranking[1]
        if p2 == pts:
            obs = f"EMPATE com {d2} ({p2} pt)"
        elif pts - p2 <= 1:
            obs = f"2ª colocada próxima: {d2} ({p2} pt)"
    return diag, pts, obs


# ---------------------------------------------------------------- Extra
PADROES_TEMPO = [
    r"\b(?:ha|faz|fazem)\s+(\w+\s+(?:dias?|semanas?|mes(?:es)?|horas?|anos?))\b",
    r"\b(desde\s+(?:ontem|anteontem|hoje|\w+))\b",
]


def extrair_duracao(frase_norm: str) -> str:
    """Extrai há quanto tempo o sintoma existe ('há dois dias', 'desde ontem')."""
    for padrao in PADROES_TEMPO:
        m = re.search(padrao, frase_norm)
        if m:
            return m.group(1)
    return "não informado"


# ---------------------------------------------------------------- Pipeline
def analisar(frases, pares) -> pd.DataFrame:
    linhas = []
    for frase in frases:
        fn = normalizar(frase)
        achados = achar_sintomas(fn, pares)
        diag, pts, obs = sugerir(pontuar(achados))
        linhas.append({
            "frase": frase,
            "sintomas_encontrados": "; ".join(achados) if achados else "-",
            "duracao": extrair_duracao(fn),
            "diagnostico_sugerido": diag,
            "pontuacao": pts,
            "observacao": obs,
        })
    return pd.DataFrame(linhas)


def mostrar(df: pd.DataFrame):
    for i, r in df.iterrows():
        print(f"\n[{i + 1}] {r['frase']}")
        print(f"    Sintomas : {r['sintomas_encontrados']}")
        print(f"    Duração  : {r['duracao']}")
        print(f"    Sugestão : {r['diagnostico_sugerido']} (pontuação {r['pontuacao']})")
        if r["observacao"]:
            print(f"    Atenção  : {r['observacao']}")


if __name__ == "__main__":
    pares = carregar_mapa()
    print(f"Mapa carregado: {len(pares)} expressões, "
          f"{len({d for _, d in pares})} doenças.")

    # --- Parte principal: as 10 frases
    resultado = analisar(carregar_frases(ARQ_FRASES), pares)
    mostrar(resultado)
    resultado.to_csv(ARQ_SAIDA, index=False, encoding="utf-8-sig")
    print(f"\nTabela final salva em {ARQ_SAIDA.name}")

    # --- Seção própria: casos difíceis (enviados pela PESSOA 3)
    # A P3 escreve estas frases de propósito para testar os limites do sistema de regras.
    # Depois de rodar, o resultado deve ser comentado aqui (o que o sistema acertou/errou):
    #
    # ANÁLISE DOS CASOS DIFÍCEIS (resultado da execução):
    # [1] ACERTOU: dor no peito + suor frio (infarto) pesam mais que coração disparado
    #     (arritmia); o sistema sugere infarto e avisa que arritmia vem logo atrás.
    # [2] ERROU (negação): "não sinto dor no peito nem batedeira" foi lido como se o
    #     paciente tivesse os sintomas. É a limitação mais grave, pois transforma um
    #     relato sem sintomas em suspeita de infarto/arritmia (empate 1 x 1).
    # [3] ACERTOU em não chutar: "gastura no peito" e "coração vai sair pela boca" são
    #     gírias fora do mapa, então encaminhou para avaliação em vez de inventar.
    #     Mas um sintoma real passou batido (falso negativo de vocabulário).
    # [4] ACERTOU PARCIALMENTE: "dor no pieto" (erro de digitação) não foi reconhecida;
    #     só "suor frio" pontuou. Sugeriu infarto, mas com apenas 1 ponto e sem o
    #     sintoma principal; o resultado ficou certo por sorte, não por robustez.
    # [5] ACERTOU: dois sintomas de insuficiência cardíaca vencem a trombose (só 1 ponto,
    #     "perna inchada"); a 2ª colocada é sinalizada.
    # Conclusão: o sistema vai bem com vocabulário conhecido, mas falha em negação,
    # gírias e erros de digitação. Melhorias possíveis: detectar "não/nem/sem" antes do
    # sintoma, aumentar o mapa e usar comparação aproximada de texto (fuzzy matching).
    print("\n" + "=" * 70 + "\nCASOS DIFÍCEIS\n" + "=" * 70)
    if ARQ_DIFICEIS.exists():
        dificeis = analisar(carregar_frases(ARQ_DIFICEIS), pares)
        mostrar(dificeis)
        dificeis.to_csv(ARQ_SAIDA_DIFICEIS, index=False, encoding="utf-8-sig")
    else:
        print("casos_dificeis.txt ainda não existe (aguardando a P3).")

    # Limitações conhecidas:
    # - Negação: "não sinto dor no peito" ainda casa com "dor no peito".
    # - Só reconhece expressões que estão no mapa (sinônimos novos passam batido).
    # - Erros de digitação não são tratados.