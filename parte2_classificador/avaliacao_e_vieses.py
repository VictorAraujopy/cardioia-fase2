
from pathlib import Path

import pandas as pd
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
SEMENTE = 42
MODELO_FINAL = "Regressão Logística"

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

TESTES_PROVOCATIVOS = [
    {"teste": "vídeo: original",
     "frase": FRASE_VIDEO,
     "esperado": ALTO},
    {"teste": "vídeo: com negação",
     "frase": "Há dois dias não sinto aperto no peito nem nada no braço quando subo escada.",
     "esperado": BAIXO},
    {"teste": "vídeo: com gíria",
     "frase": "Há dois dias sinto uma gastura no peito que corre pro braço quando subo escada.",
     "esperado": ALTO},
    {"teste": "vídeo: com 'depois de'",
     "frase": "Há dois dias sinto um aperto no peito que vai pro braço depois de subir escada.",
     "esperado": ALTO},
    {"teste": "vídeo: erro de digitação",
     "frase": "Há dois dias sinto um aperto no pieto que vai pro braço quando subo escada.",
     "esperado": ALTO},
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
    print()
    print("=" * 100)
    print(texto)
    print("=" * 100)


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
    return ", ".join(f"'{palavra}' ({valor:+.2f})" for palavra, valor in mais_fortes.items())


def carregar_dados():
    dataset = pd.read_csv(CAMINHO_DATASET, encoding="utf-8")
    frases_cegas = pd.read_csv(CAMINHO_FRASES_CEGAS, encoding="utf-8")

    assert set(dataset["situacao"]) == {ALTO, BAIXO}, "rótulo fora do padrão no dataset"
    assert set(frases_cegas["situacao"]) == {ALTO, BAIXO}, "rótulo fora do padrão nas frases cegas"
    frases_do_dataset = set(dataset["frase"].str.lower())
    assert not frases_cegas["frase"].str.lower().isin(frases_do_dataset).any(), "frase cega está no dataset"

    print(f"Dataset da P2:      {len(dataset)} frases {dataset['situacao'].value_counts().to_dict()}")
    print(f"Frases cegas da P3: {len(frases_cegas)} frases {frases_cegas['situacao'].value_counts().to_dict()}")
    return dataset, frases_cegas


def comparar_modelos(dataset):
    titulo("a) Comparação de 3 modelos · validação cruzada com 5 folds")

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
    tabela = pd.DataFrame(linhas).sort_values("acuracia_media", ascending=False)

    print(f"{'modelo':<24}{'acurácia (média ± desvio)':<30}recall de alto risco (média ± desvio)")
    for linha in tabela.itertuples():
        acuracia = f"{linha.acuracia_media:.1%} ± {linha.acuracia_desvio:.1%}"
        recall = f"{linha.recall_alto_media:.1%} ± {linha.recall_alto_desvio:.1%}"
        print(f"{linha.modelo:<24}{acuracia:<30}{recall}")

    primeiro = tabela.iloc[0]
    segundo = tabela.iloc[1]
    diferenca = primeiro.acuracia_media - segundo.acuracia_media
    maior_desvio = max(primeiro.acuracia_desvio, segundo.acuracia_desvio)
    print(f"\nDiferença entre {primeiro.modelo} e {segundo.modelo}: {diferenca:.1%} "
          f"(o resultado balança até {maior_desvio:.1%} entre os folds)")
    if diferenca < maior_desvio:
        print("→ Empate técnico: a diferença é menor que o desvio, pode ser sorte da divisão.")
    print(f"→ Modelo final: {MODELO_FINAL}. É o modelo da P2 e o único dos três que mostra o peso de cada palavra.")


def conferir_com_o_notebook_da_p2(dataset):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dataset["frase"], dataset["situacao"], test_size=0.25, stratify=dataset["situacao"], random_state=SEMENTE)
    modelo = criar_pipeline(LogisticRegression()).fit(X_treino, y_treino)
    previsto = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsto)
    recall_alto = recall_score(y_teste, previsto, pos_label=ALTO)
    print(f"\nConferência com o notebook da P2 (mesmo split 75/25): acurácia {acuracia:.1%}, "
          f"recall de alto risco {recall_alto:.1%} (no notebook: 86,8% e 89,5%)")


def rodar_frases_cegas(dataset, frases_cegas):
    titulo(f"b) Frases cegas · {MODELO_FINAL} treinada com as {len(dataset)} frases, rodada uma única vez")

    modelo_final = criar_pipeline(criar_os_tres_modelos()[MODELO_FINAL])
    modelo_final.fit(dataset["frase"], dataset["situacao"])

    resultado = frases_cegas.rename(columns={"situacao": "real"})
    resultado["previsto"] = modelo_final.predict(resultado["frase"])
    resultado["prob_alto"] = probabilidade_de_alto_risco(modelo_final, resultado["frase"])

    acuracia = accuracy_score(resultado["real"], resultado["previsto"])
    recall_alto = recall_score(resultado["real"], resultado["previsto"], pos_label=ALTO)
    falsos_negativos = resultado[(resultado["real"] == ALTO) & (resultado["previsto"] == BAIXO)]
    falsos_positivos = resultado[(resultado["real"] == BAIXO) & (resultado["previsto"] == ALTO)]

    print(f"Acurácia: {acuracia:.1%} | recall de alto risco: {recall_alto:.1%}")
    print(f"Falsos negativos (grave classificado como leve): {len(falsos_negativos)} | "
          f"falsos positivos (leve classificado como grave): {len(falsos_positivos)}")

    menor_prob = resultado["prob_alto"].min()
    maior_prob = resultado["prob_alto"].max()
    perto_do_meio = resultado["prob_alto"].between(0.4, 0.6).sum()
    print(f"Confiança: a prob. de alto risco ficou entre {menor_prob:.2f} e {maior_prob:.2f}, e "
          f"{perto_do_meio} de {len(resultado)} frases ficaram entre 0.40 e 0.60 (quase cara ou coroa)")

    erros = resultado[resultado["real"] != resultado["previsto"]]
    for erro in erros.itertuples():
        print(f"\n  ERRO | real: {erro.real} | previsto: {erro.previsto} (prob. alto {erro.prob_alto:.2f})")
        print(f"  frase: {erro.frase}")
        print(f"  palavras que mais pesaram: {palavras_que_mais_pesaram(modelo_final, erro.frase)}")

    return modelo_final


def mostrar_palavras_que_mais_pesam(modelo_final):
    titulo("c1) As 10 palavras que mais pesam para cada lado")

    pesos = pesos_para_alto_risco(modelo_final).sort_values()
    mais_puxam_para_alto = pesos.tail(10)[::-1]
    mais_puxam_para_baixo = pesos.head(10)

    print(f"{'puxam para ALTO risco':<34}puxam para BAIXO risco")
    for (palavra_alto, peso_alto), (palavra_baixo, peso_baixo) in zip(mais_puxam_para_alto.items(),
                                                                      mais_puxam_para_baixo.items()):
        print(f"  {palavra_alto:<22}{peso_alto:+.2f}{'':<6}  {palavra_baixo:<22}{peso_baixo:+.2f}")


def checar_palavras_que_nao_deveriam_pesar(modelo_final):
    titulo("c2) Palavras que não deveriam pesar (tempo, contexto, pessoa, gênero)")

    pesos = pesos_para_alto_risco(modelo_final)
    for palavra in PALAVRAS_QUE_NAO_DEVERIAM_PESAR:
        if palavra in pesos.index:
            print(f"  {palavra:<12}{pesos[palavra]:+.2f}")
        else:
            print(f"  {palavra:<12}fora do vocabulário (nenhuma frase do dataset usa)")


def testar_troca_de_genero(modelo_final):
    titulo("c3) Mesma frase, só muda o gênero")

    for par in TROCA_DE_GENERO:
        for genero in ["masculino", "feminino"]:
            previsto, prob_alto = resposta_do_modelo(modelo_final, par[genero])
            print(f"  {genero:<10}| {previsto:<12}(prob. alto {prob_alto:.2f}) | {par[genero]}")
        print()


def rodar_testes_provocativos(modelo_final):
    titulo("c4) Testes provocativos")

    quantos_enganaram = 0
    for teste in TESTES_PROVOCATIVOS:
        previsto, prob_alto = resposta_do_modelo(modelo_final, teste["frase"])
        acertou = previsto == teste["esperado"]
        if not acertou:
            quantos_enganaram += 1

        situacao = "ok   " if acertou else "ERROU"
        print(f"  {situacao} | {teste['teste']:<26}| esperado: {teste['esperado']:<12}| "
              f"previsto: {previsto:<12}(prob. alto {prob_alto:.2f})")
        print(f"          {teste['frase']}")
        if not acertou:
            print(f"          palavras que mais pesaram: {palavras_que_mais_pesaram(modelo_final, teste['frase'])}")

    print(f"\n{quantos_enganaram} de {len(TESTES_PROVOCATIVOS)} testes provocativos enganaram o modelo.")


def main():
    dataset, frases_cegas = carregar_dados()

    comparar_modelos(dataset)
    conferir_com_o_notebook_da_p2(dataset)

    modelo_final = rodar_frases_cegas(dataset, frases_cegas)

    mostrar_palavras_que_mais_pesam(modelo_final)
    checar_palavras_que_nao_deveriam_pesar(modelo_final)
    testar_troca_de_genero(modelo_final)
    rodar_testes_provocativos(modelo_final)


if __name__ == "__main__":
    main()
