
import sys
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PASTA_DO_SCRIPT = Path(__file__).parent
CAMINHO_DATASET = PASTA_DO_SCRIPT / "frases_risco.csv"
CAMINHO_FRASES_CEGAS = PASTA_DO_SCRIPT / "frases_teste_cegas.csv"

ALTO = "alto risco"
BAIXO = "baixo risco"
COR_DO_RISCO = {ALTO: "red", BAIXO: "green"}
SEMENTE = 42
MODELO_FINAL = "Regressão Logística"

PAUSAR_ENTRE_PARTES = "--pausar" in sys.argv
console = Console()

FRASE_VIDEO = "Há dois dias sinto um aperto no peito que vai pro braço quando subo escada."

PALAVRAS_QUE_NAO_DEVERIAM_PESAR = [
    "ontem", "hoje", "trabalho",
    "mulher", "homem", "idoso", "idosa",
    "cansado", "cansada", "tonto", "tonta",
]

TROCA_DE_GENERO = [
    {"masculino": "Estou tonto e com o coração acelerado.",
     "feminino": "Estou tonta e com o coração acelerado."},
    {"masculino": "Fiquei nervoso e o coração disparou.",
     "feminino": "Fiquei nervosa e o coração disparou."},
    {"masculino": "Acordei sufocado de madrugada e o coração tava disparado.",
     "feminino": "Acordei sufocada de madrugada e o coração tava disparado."},
]

TESTES_COM_A_FRASE_DO_VIDEO = [
    {"teste": "original",
     "frase": FRASE_VIDEO,
     "esperado": ALTO},
    {"teste": "com negação",
     "frase": "Há dois dias não sinto aperto no peito nem nada no braço quando subo escada.",
     "esperado": BAIXO},
    {"teste": "com gíria",
     "frase": "Há dois dias sinto uma gastura no peito que corre pro braço quando subo escada.",
     "esperado": ALTO},
    {"teste": "com 'depois de'",
     "frase": "Há dois dias sinto um aperto no peito que vai pro braço depois de subir escada.",
     "esperado": ALTO},
    {"teste": "com erro de digitação",
     "frase": "Há dois dias sinto um aperto no pieto que vai pro braço quando subo escada.",
     "esperado": ALTO},
]

OUTROS_TESTES_PROVOCATIVOS = [
    {"teste": "negação simples",
     "frase": "Não sinto dor no peito.",
     "esperado": BAIXO},
    {"teste": "negação sem 'não'",
     "frase": "Sem dor nenhuma, tô bem.",
     "esperado": BAIXO},
    {"teste": "sinônimo fora do treino",
     "frase": "Sinto uma opressão no tórax que desce pelo braço.",
     "esperado": ALTO},
    {"teste": "frase longa",
     "frase": "Ontem fui no mercado, depois passei na farmácia, depois busquei meu filho na escola "
              "e depois de tudo isso senti uma dor forte no peito que foi pro braço esquerdo e não passou até agora.",
     "esperado": ALTO},
]


def titulo(texto):
    console.print()
    console.rule(f"[bold cyan]{texto}")
    console.print()


def pausa():
    """Com --pausar, espera um Enter e limpa a tela antes da próxima parte (bom para apresentar)."""
    if PAUSAR_ENTRE_PARTES:
        console.input("\n[dim]Enter para continuar[/dim] ")
        console.clear()


def risco_colorido(rotulo):
    cor = COR_DO_RISCO[rotulo]
    return f"[{cor}]{rotulo}[/{cor}]"


def criar_pipeline(classificador):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
        ("clf", classificador),
    ])


def criar_os_tres_modelos():
    return {
        "Regressão Logística": LogisticRegression(),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=SEMENTE),
        "Naive Bayes": MultinomialNB(),
    }


def probabilidade_de_alto_risco(modelo, frases):
    coluna_do_alto = list(modelo.classes_).index(ALTO)
    return modelo.predict_proba(frases)[:, coluna_do_alto]


def resposta_do_modelo(modelo, frase):
    previsto = modelo.predict([frase])[0]
    prob_alto = probabilidade_de_alto_risco(modelo, [frase])[0]
    return previsto, prob_alto


def pesos_para_alto_risco(modelo):
    """Peso de cada palavra na Regressão Logística: positivo puxa para alto risco, negativo para baixo."""
    vocabulario = modelo.named_steps["tfidf"].get_feature_names_out()
    regressao = modelo.named_steps["clf"]

    # o sklearn guarda os pesos a favor de classes_[1], que em ordem alfabética é "baixo risco"
    assert regressao.classes_[1] == BAIXO
    pesos_a_favor_de_baixo = pd.Series(regressao.coef_[0], index=vocabulario)
    return -pesos_a_favor_de_baixo


def palavras_que_mais_pesaram(modelo, frase, quantas=3):
    """As palavras desta frase que mais empurraram o modelo para a resposta que ele deu."""
    pesos = pesos_para_alto_risco(modelo)
    tfidf_da_frase = modelo.named_steps["tfidf"].transform([frase]).toarray()[0]
    contribuicao = pd.Series(tfidf_da_frase, index=pesos.index) * pesos
    contribuicao = contribuicao[contribuicao != 0]

    previsto = modelo.predict([frase])[0]
    if previsto == ALTO:
        mais_fortes = contribuicao.nlargest(quantas)
    else:
        mais_fortes = contribuicao.nsmallest(quantas)
    return ", ".join(f"'{palavra}'" for palavra in mais_fortes.index)


def carregar_dados():
    dataset = pd.read_csv(CAMINHO_DATASET, encoding="utf-8")
    frases_cegas = pd.read_csv(CAMINHO_FRASES_CEGAS, encoding="utf-8")

    assert set(dataset["situacao"]) == {ALTO, BAIXO}, "rótulo fora do padrão no dataset"
    assert set(frases_cegas["situacao"]) == {ALTO, BAIXO}, "rótulo fora do padrão nas frases cegas"
    frases_do_dataset = set(dataset["frase"].str.lower())
    assert not frases_cegas["frase"].str.lower().isin(frases_do_dataset).any(), "frase cega está no dataset"

    contagem_dataset = dataset["situacao"].value_counts()
    contagem_cegas = frases_cegas["situacao"].value_counts()
    console.print(f"Dataset da P2: [bold]{len(dataset)}[/bold] frases "
                  f"({contagem_dataset[ALTO]} alto, {contagem_dataset[BAIXO]} baixo)")
    console.print(f"Frases cegas da P3: [bold]{len(frases_cegas)}[/bold] frases "
                  f"({contagem_cegas[ALTO]} alto, {contagem_cegas[BAIXO]} baixo)")
    return dataset, frases_cegas


def comparar_modelos(dataset):
    titulo("a) Comparação de 3 modelos")

    # o CSV vem ordenado (75 alto e depois 75 baixo); embaralhar evita folds com blocos de frases parecidas
    cinco_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    metricas = {
        "acuracia": "accuracy",
        "recall_alto": make_scorer(recall_score, pos_label=ALTO),
    }

    linhas = []
    for nome, classificador in criar_os_tres_modelos().items():
        notas = cross_validate(criar_pipeline(classificador), dataset["frase"], dataset["situacao"],
                               cv=cinco_folds, scoring=metricas)
        linhas.append({
            "modelo": nome,
            "acuracia_media": notas["test_acuracia"].mean(),
            "acuracia_desvio": notas["test_acuracia"].std(),
            "recall_alto_media": notas["test_recall_alto"].mean(),
            "recall_alto_desvio": notas["test_recall_alto"].std(),
        })
    resultados = pd.DataFrame(linhas).sort_values("acuracia_media", ascending=False)

    tabela = Table(title="Validação cruzada com 5 folds (média ± desvio)")
    tabela.add_column("Modelo")
    tabela.add_column("Acurácia", justify="right")
    tabela.add_column("Recall de alto risco", justify="right")
    for linha in resultados.itertuples():
        tabela.add_row(linha.modelo,
                       f"{linha.acuracia_media:.1%} ± {linha.acuracia_desvio:.1%}",
                       f"{linha.recall_alto_media:.1%} ± {linha.recall_alto_desvio:.1%}")
    console.print(tabela)
    console.print()

    primeiro = resultados.iloc[0]
    segundo = resultados.iloc[1]
    diferenca = primeiro.acuracia_media - segundo.acuracia_media
    maior_desvio = max(primeiro.acuracia_desvio, segundo.acuracia_desvio)
    if diferenca < maior_desvio:
        console.print(f"[bold yellow]→ Empate técnico:[/bold yellow] a diferença ({diferenca:.1%}) "
                      f"é menor que o quanto o resultado balança ({maior_desvio:.1%})")
        console.print()
    console.print(f"[bold green]→ Modelo final: {MODELO_FINAL}[/bold green], "
                  f"o único dos três que mostra o peso de cada palavra")


def conferir_com_o_notebook_da_p2(dataset):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dataset["frase"], dataset["situacao"], test_size=0.25, stratify=dataset["situacao"], random_state=SEMENTE)
    modelo = criar_pipeline(LogisticRegression()).fit(X_treino, y_treino)
    previsto = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsto)
    recall_alto = recall_score(y_teste, previsto, pos_label=ALTO)
    console.print(f"[dim]Conferência no split da P2 (75/25): acurácia {acuracia:.1%}, "
                  f"recall {recall_alto:.1%} (notebook: 86,8% e 89,5%)[/dim]")


def rodar_frases_cegas(dataset, frases_cegas):
    titulo(f"b) Frases cegas · {MODELO_FINAL} rodada uma única vez")

    modelo_final = criar_pipeline(criar_os_tres_modelos()[MODELO_FINAL])
    modelo_final.fit(dataset["frase"], dataset["situacao"])

    resultado = frases_cegas.rename(columns={"situacao": "real"})
    resultado["previsto"] = modelo_final.predict(resultado["frase"])
    resultado["prob_alto"] = probabilidade_de_alto_risco(modelo_final, resultado["frase"])

    acertos = (resultado["real"] == resultado["previsto"]).sum()
    casos_graves = resultado[resultado["real"] == ALTO]
    graves_pegos = (casos_graves["previsto"] == ALTO).sum()
    falsos_negativos = resultado[(resultado["real"] == ALTO) & (resultado["previsto"] == BAIXO)]
    falsos_positivos = resultado[(resultado["real"] == BAIXO) & (resultado["previsto"] == ALTO)]
    acuracia = accuracy_score(resultado["real"], resultado["previsto"])
    recall_alto = recall_score(resultado["real"], resultado["previsto"], pos_label=ALTO)

    tabela = Table(title=f"{len(resultado)} frases que o modelo nunca viu")
    tabela.add_column("Métrica")
    tabela.add_column("Valor", justify="right")
    tabela.add_column("")
    tabela.add_row("Acurácia", f"[bold]{acuracia:.1%}[/bold]", f"{acertos} de {len(resultado)}")
    tabela.add_row("Recall de alto risco", f"[bold]{recall_alto:.1%}[/bold]", f"{graves_pegos} de {len(casos_graves)}")
    tabela.add_row("Falsos negativos", str(len(falsos_negativos)), "grave classificado como leve")
    tabela.add_row("Falsos positivos", str(len(falsos_positivos)), "leve classificado como grave")
    console.print(tabela)

    return modelo_final, resultado


def regua_de_probabilidade(prob_alto, cor, metade=20):
    """Desenha a probabilidade numa régua de 0 a 1: à esquerda do │ é baixo risco, à direita é alto risco."""
    lado_baixo = ["─"] * metade
    lado_alto = ["─"] * metade
    bolinha = f"[bold {cor}]●[/bold {cor}]"
    if prob_alto < 0.5:
        lado_baixo[int(prob_alto / 0.5 * metade)] = bolinha
    else:
        lado_alto[min(int((prob_alto - 0.5) / 0.5 * metade), metade - 1)] = bolinha
    return "".join(lado_baixo) + "│" + "".join(lado_alto)


def mostrar_certeza_do_modelo(resultado):
    titulo("b) Frases cegas · quanto o modelo tem certeza?")

    console.print("O modelo dá a cada frase uma [bold]probabilidade de alto risco[/bold], de 0 a 1. "
                  "De [bold]0.50[/bold] pra cima, ele responde alto risco.")

    tabela = Table(box=box.SIMPLE)
    tabela.add_column("Prob.", justify="right", no_wrap=True)
    tabela.add_column("0" + " " * 18 + "0.5" + " " * 18 + "1", no_wrap=True, min_width=41)
    tabela.add_column("Resposta certa", no_wrap=True)
    tabela.add_column("Modelo disse", no_wrap=True)
    tabela.add_column("", no_wrap=True)

    da_menor_para_a_maior = resultado.sort_values("prob_alto")
    abaixo_do_limite = da_menor_para_a_maior[da_menor_para_a_maior["prob_alto"] < 0.5]
    ultima_antes_do_limite = abaixo_do_limite.index[-1]

    for linha in da_menor_para_a_maior.itertuples():
        marca = "[green]✓[/green]" if linha.real == linha.previsto else "[bold red]✗[/bold red]"
        tabela.add_row(f"{linha.prob_alto:.2f}",
                       regua_de_probabilidade(linha.prob_alto, COR_DO_RISCO[linha.previsto]),
                       risco_colorido(linha.real), risco_colorido(linha.previsto), marca,
                       end_section=(linha.Index == ultima_antes_do_limite))
    console.print(tabela)

    acertos = (resultado["real"] == resultado["previsto"]).sum()
    perto_do_limite = resultado["prob_alto"].between(0.4, 0.6).sum()
    console.print(f"Acertou [bold]{acertos} de {len(resultado)}[/bold], mas por pouco: "
                  f"[bold yellow]{perto_do_limite} de {len(resultado)}[/bold yellow] ficaram entre 0.40 e 0.60, "
                  f"coladas no limite.")
    console.print("Uma palavra a mais ou a menos já muda a resposta (ver c2).")


def mostrar_palavras_que_mais_pesam(modelo_final):
    titulo("c1) As 10 palavras que mais pesam para cada lado")

    pesos = pesos_para_alto_risco(modelo_final).sort_values()
    mais_puxam_para_alto = pesos.tail(10)[::-1]
    mais_puxam_para_baixo = pesos.head(10)

    tabela = Table(title="Peso de cada palavra no modelo final")
    tabela.add_column("Puxam para ALTO risco", style="red")
    tabela.add_column("Peso", justify="right")
    tabela.add_column("Puxam para BAIXO risco", style="green")
    tabela.add_column("Peso", justify="right")
    for (palavra_alto, peso_alto), (palavra_baixo, peso_baixo) in zip(mais_puxam_para_alto.items(),
                                                                      mais_puxam_para_baixo.items()):
        tabela.add_row(palavra_alto, f"{peso_alto:+.2f}", palavra_baixo, f"{peso_baixo:+.2f}")
    console.print(tabela)


def rodar_testes(titulo_da_parte, testes, modelo_final):
    titulo(titulo_da_parte)

    tabela = Table(show_lines=True)
    tabela.add_column("")
    tabela.add_column("Teste e frase")
    tabela.add_column("Esperado")
    tabela.add_column("Modelo disse")

    quantos_enganaram = 0
    for teste in testes:
        previsto, prob_alto = resposta_do_modelo(modelo_final, teste["frase"])
        acertou = previsto == teste["esperado"]

        descricao = f"[bold]{teste['teste']}[/bold]\n{teste['frase']}"
        if acertou:
            marca = "[green]✓[/green]"
        else:
            marca = "[bold red]✗[/bold red]"
            descricao += f"\n[yellow]pesaram: {palavras_que_mais_pesaram(modelo_final, teste['frase'])}[/yellow]"
            quantos_enganaram += 1

        tabela.add_row(marca, descricao, risco_colorido(teste["esperado"]),
                       f"{risco_colorido(previsto)} ({prob_alto:.2f})")
    console.print(tabela)

    console.print(f"\n[bold]{quantos_enganaram} de {len(testes)}[/bold] testes enganaram o modelo.")


def testar_troca_de_genero(modelo_final):
    titulo("c4) Mesma frase, só muda o gênero")

    tabela = Table()
    tabela.add_column("Gênero")
    tabela.add_column("Frase")
    tabela.add_column("Modelo disse")
    for par in TROCA_DE_GENERO:
        for genero in ["masculino", "feminino"]:
            previsto, prob_alto = resposta_do_modelo(modelo_final, par[genero])
            tabela.add_row(genero, par[genero], f"{risco_colorido(previsto)} ({prob_alto:.2f})",
                           end_section=(genero == "feminino"))
    console.print(tabela)


def checar_palavras_que_nao_deveriam_pesar(modelo_final):
    titulo("c5) Palavras que não deveriam pesar (tempo, contexto, pessoa, gênero)")

    pesos = pesos_para_alto_risco(modelo_final)
    tabela = Table()
    tabela.add_column("Palavra")
    tabela.add_column("Peso", justify="right")
    for palavra in PALAVRAS_QUE_NAO_DEVERIAM_PESAR:
        if palavra in pesos.index:
            tabela.add_row(palavra, f"{pesos[palavra]:+.2f}")
        else:
            tabela.add_row(palavra, "[dim]fora do vocabulário[/dim]")
    console.print(tabela)


def main():
    dataset, frases_cegas = carregar_dados()
    conferir_com_o_notebook_da_p2(dataset)

    comparar_modelos(dataset)
    pausa()

    modelo_final, resultado_das_cegas = rodar_frases_cegas(dataset, frases_cegas)
    pausa()

    mostrar_certeza_do_modelo(resultado_das_cegas)
    pausa()

    mostrar_palavras_que_mais_pesam(modelo_final)
    pausa()

    rodar_testes(f"c2) Mesma queixa do paciente, dita de {len(TESTES_COM_A_FRASE_DO_VIDEO)} jeitos",
                 TESTES_COM_A_FRASE_DO_VIDEO, modelo_final)
    pausa()

    rodar_testes("c3) Outros testes provocativos", OUTROS_TESTES_PROVOCATIVOS, modelo_final)
    pausa()

    testar_troca_de_genero(modelo_final)
    pausa()

    checar_palavras_que_nao_deveriam_pesar(modelo_final)


if __name__ == "__main__":
    main()
