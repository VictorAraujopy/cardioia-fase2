# CardioIA — Fase 2: Diagnóstico Automatizado

Módulo de apoio ao diagnóstico que lê relatos de pacientes, identifica sintomas e sugere diagnósticos (Parte 1), e classifica o nível de risco de frases com TF-IDF e Machine Learning (Parte 2).

**Vídeo de demonstração:** [Assistir no YouTube](https://youtu.be/I-MGwVa-y5c)

## Integrantes

| Nome | RM |
|---|---|
| Jacqueline Nanami Matushima | RM568498 |
| Pedro Zanon Castro Santana | RM567350 |
| Victor Araujo Ferreira da Silva | RM567619 |

## Estrutura do repositório

```
cardioia-fase2/
├── README.md
├── requirements.txt
├── .gitignore
├── parte1_extracao/
│   ├── frases_sintomas.txt         # 10 relatos de pacientes
│   ├── mapa_conhecimento.csv       # mapa sintoma → doença
│   ├── casos_dificeis.txt          # frases difíceis para testar a Parte 1
│   └── extracao_diagnostico.py     # extração de sintomas e sugestão de diagnóstico
└── parte2_classificador/
    ├── frases_risco.csv            # 150 frases rotuladas (alto/baixo risco)
    ├── classificador_risco.ipynb   # TF-IDF + Regressão Logística
    ├── frases_teste_cegas.csv      # 20 frases cegas para a avaliação final
    └── avaliacao_e_vieses.py       # comparação de modelos e caça aos vieses
```

## Como rodar

Requer **Python 3.11 ou mais novo** (testado em clone limpo com Python 3.11 e 3.14, com as versões fixadas no `requirements.txt`).

```bash
git clone https://github.com/VictorAraujopy/cardioia-fase2.git
cd cardioia-fase2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Depois, da raiz do repositório:

```bash
python parte1_extracao/extracao_diagnostico.py                    # Parte 1
jupyter notebook parte2_classificador/classificador_risco.ipynb   # Parte 2 (Kernel → Restart & Run All)
python parte2_classificador/avaliacao_e_vieses.py                 # Avaliação dos modelos e vieses
```

---

## Parte 1 — Extração de sintomas e sugestão de diagnóstico

Sistema **baseado em regras** (sem machine learning): lê relatos de pacientes, procura sintomas conhecidos e sugere a doença mais provável.

### Arquivos (`parte1_extracao/`)

| Arquivo | Descrição |
|---|---|
| `frases_sintomas.txt` | 10 relatos simulados, com o que o paciente sente, quando começou e como afeta a rotina |
| `mapa_conhecimento.csv` | Mapa sintoma → doença (36 linhas, 9 doenças, colunas `sintoma_1, sintoma_2, doenca_associada`) |
| `extracao_diagnostico.py` | Código que lê as frases, identifica sintomas e sugere diagnóstico |
| `casos_dificeis.txt` | Frases difíceis para testar os limites do sistema |
| `resultados.csv` | Tabela final gerada pelo código (criada ao rodar o script) |

Doenças cobertas: infarto agudo do miocárdio, angina, insuficiência cardíaca, arritmia, hipertensão arterial, pericardite, miocardite, estenose aórtica e trombose venosa profunda.

### Como funciona

1. Lê as frases (`.txt`) e o mapa de conhecimento (`.csv`).
2. Normaliza o texto: minúsculas, sem acento e sem pontuação (assim "tórax" e "torax" são iguais).
3. Procura cada expressão do mapa na frase com regex e `\b` (limite de palavra), para não casar pedaço de palavra.
4. Pontua: cada sintoma encontrado soma +1 para a doença associada, e a doença com mais pontos é a sugestão. Empates ou uma 2ª colocada próxima são sinalizados.
5. Sem nenhum sintoma reconhecido, o sistema responde "nenhum sintoma reconhecido, encaminhar para avaliação" em vez de chutar um diagnóstico.
6. Extra: extrai também há quanto tempo o sintoma existe ("há dois dias", "desde ontem").
7. Salva a tabela final (frase, sintomas, duração, diagnóstico, pontuação) em `resultados.csv`.

### Como rodar

```bash
cd parte1_extracao
python extracao_diagnostico.py
```

### Exemplo de saída

```
[1] Há dois dias sinto um aperto no peito que vai pro braço quando subo escada.
    Sintomas : aperto no peito; quando subo escada; vai pro braco
    Duração  : dois dias
    Sugestão : Infarto agudo do miocárdio (pontuação 2)
    Atenção  : 2ª colocada próxima: Angina (1 pt)
```

### Limitações

- **Negação:** "não sinto dor no peito" ainda casa com "dor no peito".
- **Vocabulário limitado:** só reconhece expressões que estão no mapa; sinônimos novos passam despercebidos.
- **Erros de digitação** não são tratados.
- **Empates:** quando duas doenças empatam, o código mostra o empate na coluna de observação, mas a sugestão principal segue a ordem alfabética.

### Fontes consultadas

- Sociedade Brasileira de Cardiologia (SBC). Diretrizes e materiais para pacientes. https://www.portal.cardiol.br
- Manual MSD: Distúrbios cardiovasculares. https://www.msdmanuals.com/pt/profissional/distúrbios-cardiovasculares

---

## Parte 2 — Classificador de risco

Modelo de machine learning que lê a frase de um paciente e classifica como **alto risco** ou **baixo risco**, simulando uma triagem automática.

### Arquivos (`parte2_classificador/`)

| Arquivo | Descrição |
|---|---|
| `frases_risco.csv` | 150 frases rotuladas (colunas `frase` e `situacao`) |
| `classificador_risco.ipynb` | TF-IDF, treino, avaliação e análise de padrões e distorções |

### Dataset

150 frases inventadas pelo grupo, 75 de alto risco e 75 de baixo risco. Inclui linguagem formal e popular ("batedeira", "tô sem ar", "dor no pieto") e pares armadilha, com a mesma palavra em contextos de risco diferente ("dor no peito leve depois de comer muito" × "dor no peito forte que vai para o braço esquerdo"). A frase do paciente do vídeo não está no dataset.

### Critério de rotulagem

- **alto risco:** dor ou aperto no peito forte, prolongado ou que surge com esforço; dor que irradia para braço, mandíbula, pescoço ou costas; falta de ar em repouso ou ao deitar; desmaio ou quase desmaio; palpitação com tontura, dor ou falta de ar; sintoma novo e intenso que impede a rotina.
- **baixo risco:** sintoma leve e passageiro, com causa aparente (exercício pesado, refeição pesada, café, ansiedade pontual, má postura), sem sinal de alarme e sem prejuízo importante da rotina.
- **desempate:** havendo qualquer sinal de alarme, a frase é alto risco.

### Modelo

TF-IDF com palavras e pares de palavras (`ngram_range=(1, 2)`), sem remover stopwords, para manter "não" e "sem", seguido de Regressão Logística. Os dois ficam dentro de um `Pipeline` para evitar vazamento de dados. Divisão 75% treino / 25% teste, estratificada, com `random_state=42`.

### Como rodar

```bash
jupyter notebook parte2_classificador/classificador_risco.ipynb
```

Depois, use Kernel → Restart & Run All. O notebook encontra o CSV rodando de dentro da pasta ou da raiz do repositório.

### Resultados no teste (38 frases)

| Métrica | Valor |
|---|---|
| Acurácia | 86,8% |
| Recall de alto risco | 89,5% (17 de 19) |
| Falsos negativos | 2 |
| Falsos positivos | 3 |

O recall de alto risco é a métrica mais importante, porque um falso negativo manda um paciente grave para o fim da fila.

### Padrões e distorções

- O modelo associa "dor no peito" a gravidade, mas perdeu os 2 casos de desmaio sem dor no peito.
- Palavras sem sentido clínico, como "depois" e "de", são as que mais pesam para baixo risco, porque refletem o jeito como escrevemos as frases.
- Frases com negação ("não sinto dor no peito") são classificadas como alto risco.
- A frase do vídeo foi classificada como alto risco, mas com probabilidade de apenas 0,59.

Os números são otimistas, porque treino e teste foram escritos pelas mesmas pessoas. O teste com frases cegas, na seção de avaliação, mede o desempenho de forma mais honesta.

---

## Parte 3 — Avaliação dos modelos, governança e vieses

O notebook da Parte 2 treina e testa um modelo. Esta seção responde outra pergunta: **quanto dá para confiar nele?** Tudo aqui é gerado por `parte2_classificador/avaliacao_e_vieses.py`, com o mesmo CSV, o mesmo TF-IDF e o mesmo split do notebook. O script confere isso: no split da Parte 2, ele reproduz os 86,8% de acurácia e 89,5% de recall.

### Arquivos (`parte2_classificador/`)

| Arquivo | Descrição |
|---|---|
| `frases_teste_cegas.csv` | 20 frases escritas sem ver o dataset, usadas uma única vez na avaliação final |
| `avaliacao_e_vieses.py` | Comparação de 3 modelos, teste nas frases cegas e caça aos vieses |

### Como rodar

```bash
python parte2_classificador/avaliacao_e_vieses.py             # mostra tudo de uma vez
python parte2_classificador/avaliacao_e_vieses.py --pausar    # uma parte por tela, Enter avança
```

### Comparação de 3 modelos

Validação cruzada com 5 folds: cada modelo é treinado e testado 5 vezes, cada vez com um pedaço diferente do dataset como teste. A média é mais confiável que um único split, e o desvio mostra o quanto o resultado balança. Os folds são embaralhados (`shuffle=True, random_state=42`) porque o CSV vem ordenado, com as 75 frases de alto risco antes das 75 de baixo risco.

| Modelo | Acurácia (média ± desvio) | Recall de alto risco (média ± desvio) |
|---|---|---|
| Regressão Logística | 88,7% ± 7,8% | 88,0% ± 7,8% |
| Naive Bayes | 87,3% ± 3,9% | 88,0% ± 5,0% |
| Árvore de Decisão | 73,3% ± 3,0% | 72,0% ± 11,5% |

- **Regressão Logística e Naive Bayes empatam tecnicamente:** a diferença (1,3 ponto) é bem menor que o desvio (até 7,8 pontos). Com 150 frases, declarar um vencedor seria contar com a sorte da divisão.
- **A Árvore de Decisão fica para trás:** com poucas frases, ela decora regras do treino (overfitting) e generaliza mal.
- **Modelo final: Regressão Logística.** Tem a maior média, é o modelo do notebook e é o único dos três que mostra o peso de cada palavra, o que permite a caça aos vieses. A escolha foi feita pela validação cruzada, antes de usar as frases cegas.

### Teste com frases cegas

20 frases (10 de alto e 10 de baixo risco) escritas sem ver o dataset da Parte 2 e rotuladas com o mesmo critério. Incluem negação, gíria ("gastura", "batedeira", "apaguei"), sinônimo incomum ("agonia no peito", "queimação"), frases longas e armadilhas nos dois sentidos: "peito dolorido depois do supino" é baixo risco, e "acordei sem conseguir respirar" é alto risco sem ter a palavra "dor". O modelo final, treinado com as 150 frases, foi rodado nelas uma única vez.

| Métrica | Valor |
|---|---|
| Acurácia | 95% (19 de 20) |
| Recall de alto risco | 100% (10 de 10) |
| Falsos negativos | 0 |
| Falsos positivos | 1 |

**O erro:** "Não tenho dor no peito nem falta de ar, só uma tosse chata desde que peguei friagem semana passada" (baixo risco) foi classificada como alto risco (0,57). As palavras que mais pesaram foram "no peito", "dor no" e "ar": o modelo enxerga os sintomas e ignora o "não".

**Por que 95% não é motivo de festa:** a probabilidade de alto risco ficou entre 0,36 e 0,67 em todas as frases, e metade delas (10 de 20) ficou entre 0,40 e 0,60, quase cara ou coroa. O modelo acertou, mas sem convicção: uma palavra a mais ou a menos muda a resposta, como mostram os testes provocativos. E, com só 20 frases, cada erro vale 5 pontos de acurácia.

### Caça aos vieses

**Palavras que mais pesam.** Para alto risco: "peito" (+0,82), "no peito", "forte", "dor no" e "ar". Para baixo risco: "depois" (−1,35), "depois de" (−1,12), "de" (−1,08), "um" (−0,86) e "leve" (−0,84). Palavras sem sentido clínico ("depois", "de", "um", "para", "que") estão entre as que mais pesam. O modelo aprendeu o nosso jeito de escrever: as frases de baixo risco quase sempre explicam a causa com "depois de comer" ou "depois do treino".

**Gênero.** O modelo trata "tonto" e "tonta" como palavras diferentes: "tonto" puxa para alto risco (+0,29) e "tonta", para baixo (−0,10), porque "tonto" aparece 3 vezes no dataset e "tonta", 1.

| Frase | Previsão (prob. de alto risco) |
|---|---|
| Estou tonto e com o coração acelerado. | alto (0,56) |
| Estou tonta e com o coração acelerado. | alto (0,51) |
| Fiquei nervoso e o coração disparou. | baixo (0,49) |
| Fiquei nervosa e o coração disparou. | alto (0,50) |

O efeito é pequeno, mas a resposta muda só pelo gênero de quem fala. Num par em que nenhuma das duas formas aparece no dataset ("sufocado" e "sufocada"), a probabilidade é igual (0,54), o que confirma que a causa é a frequência das palavras no treino. Já "mulher", "homem" e "idoso" nem estão no vocabulário: o modelo não sabe quem é o paciente, embora idade e sexo mudem o risco cardíaco de verdade.

**Testes provocativos** (modelo final, treinado com as 150 frases). 5 de 9 enganaram o modelo:

| Teste | Frase | Esperado | Previsto (prob. de alto) | Resultado |
|---|---|---|---|---|
| frase do vídeo | Há dois dias sinto um aperto no peito que vai pro braço quando subo escada. | alto | alto (0,58) | acertou |
| vídeo com negação | Há dois dias não sinto aperto no peito nem nada no braço quando subo escada. | baixo | alto (0,56) | **errou** |
| vídeo com gíria | ...sinto uma gastura no peito que corre pro braço quando subo escada. | alto | alto (0,61) | acertou |
| vídeo com "depois de" | ...sinto um aperto no peito que vai pro braço depois de subir escada. | alto | baixo (0,49) | **errou** |
| vídeo com erro de digitação | ...sinto um aperto no pieto que vai pro braço quando subo escada. | alto | alto (0,53) | acertou |
| negação simples | Não sinto dor no peito. | baixo | alto (0,67) | **errou** |
| negação sem "não" | Sem dor nenhuma, tô bem. | baixo | alto (0,51) | **errou** |
| sinônimo fora do treino | Sinto uma opressão no tórax que desce pelo braço. | alto | alto (0,60) | acertou |
| frase longa | Ontem fui no mercado, depois passei na farmácia, (...) e depois de tudo isso senti uma dor forte no peito que foi pro braço esquerdo e não passou até agora. | alto | baixo (0,48) | **errou** |

O caso mais claro é a frase do vídeo com "depois de". O sentido clínico é o mesmo (dor que vai para o braço ao subir escada), mas trocar "quando subo" por "depois de subir" basta para virar baixo risco. É o viés do "depois de", visto nos pesos, agindo numa frase real.

### Governança e vieses

- **Origem dos dados:** todas as frases são inventadas, escritas por 3 estudantes com vocabulário parecido. O modelo aprendeu o nosso jeito de escrever ("depois de" como sinal de baixo risco), não a medicina, e um paciente real fala de outro jeito.
- **Rótulos sem médico:** o critério de rotulagem foi definido pelo grupo a partir de sinais de alarme, sem revisão de um profissional de saúde. Qualquer erro de rótulo vira erro do modelo.
- **Erros com custos diferentes:** um falso negativo (caso grave classificado como leve) manda o paciente para o fim da fila e é muito pior que um falso positivo. Por isso olhamos o recall de alto risco, e não só a acurácia. Num sistema real, daria para baixar o limite de decisão (hoje 0,5) para pegar mais casos graves, aceitando mais falsos positivos.
- **Quem o modelo atende mal:** quem usa negação, escreve com erros, usa gíria regional ou conta a história com muitos detalhes. Na prática, isso pode prejudicar justamente pacientes com menos escolaridade ou de outras regiões. O dataset também não tem idade, sexo nem histórico, que mudam o risco cardíaco de verdade.
- **LGPD:** dado de saúde é dado pessoal sensível (Lei 13.709/2018, art. 5º, II). Um sistema real precisaria de base legal para tratá-lo (art. 11, como a tutela da saúde por profissionais e serviços de saúde), anonimização, controle de acesso e registro de quem consultou cada dado. Este projeto usa só frases inventadas, sem nenhum dado de pessoa real.
- **Papel da IA:** o modelo sugere e prioriza, e quem decide é o profissional de saúde. A LGPD garante ao paciente o direito de pedir revisão de decisões tomadas só por sistema automatizado (art. 20). Mostrar a probabilidade junto com o rótulo ajuda nisso: um "alto risco (0,51)" avisa que o modelo está em dúvida e que o caso precisa de olhar humano.

---

## Aviso

Projeto acadêmico com dados 100% simulados. **Não é uma ferramenta de diagnóstico real** e não substitui avaliação médica.
