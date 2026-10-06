# CardioIA — Fase 2: Diagnóstico Automatizado

Módulo de apoio ao diagnóstico que lê relatos de pacientes, identifica sintomas e sugere diagnósticos (Parte 1), e classifica o nível de risco de frases com TF-IDF e Machine Learning (Parte 2).

**Vídeo de demonstração:** (link do YouTube, não listado)

## Integrantes

| Nome | RM |
|---|---|
| Jacqueline Nanami Matushima | RM568498 |
| Pedro Zanon Castro Santana | RM567350 |
| Victor Araujo Ferreira da Silva | RM567619 |

## Estrutura do repositório

```
cardioia-fase2/
├── parte1_extracao/        # relatos, mapa de conhecimento e extração de sintomas
└── parte2_classificador/   # dataset rotulado, classificador TF-IDF e avaliação
```

## Como rodar

```bash
git clone https://github.com/VictorAraujopy/cardioia-fase2.git
cd cardioia-fase2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Parte 1 — Extração de sintomas e sugestão de diagnóstico

Sistema **baseado em regras** (sem machine learning): lê relatos de pacientes, procura sintomas conhecidos e sugere a doença mais provável.

### Arquivos (`parte1_extracao/`)

| Arquivo | Descrição |
|---|---|
| `frases_sintomas.txt` | 10 relatos simulados, com o que o paciente sente, quando começou e como afeta a rotina |
| `mapa_conhecimento.csv` | Mapa sintoma → doença (36 linhas, 9 doenças, colunas `sintoma_1, sintoma_2, doenca_associada`) |
| `extracao_diagnostico.py` | Código que lê as frases, identifica sintomas e sugere diagnóstico |
| `casos_dificeis.txt` | Frases difíceis para testar os limites do sistema |
| `resultados.csv` | Tabela final gerada pelo código |

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

- (citar aqui as fontes usadas para montar o mapa, ex.: Sociedade Brasileira de Cardiologia, Manual MSD)

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

- **alto risco:** sinal de alarme, como dor no peito forte ou com esforço, dor que irradia (braço, mandíbula, pescoço, costas), falta de ar em repouso ou ao deitar, desmaio ou palpitação com tontura.
- **baixo risco:** sintoma leve, passageiro e com causa aparente (exercício pesado, refeição pesada, café, ansiedade pontual), sem prejuízo importante da rotina.
- **desempate:** havendo qualquer sinal de alarme, a frase é alto risco.

### Modelo

TF-IDF com palavras e pares de palavras (`ngram_range=(1, 2)`), sem remover stopwords, para manter "não" e "sem", seguido de Regressão Logística. Os dois ficam dentro de um `Pipeline` para evitar vazamento de dados. Divisão 75% treino / 25% teste, estratificada, com `random_state=42`.

### Como rodar

    jupyter notebook parte2_classificador/classificador_risco.ipynb

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

## Avaliação dos modelos, governança e vieses

(a preencher)

## Aviso

Projeto acadêmico com dados 100% simulados. **Não é uma ferramenta de diagnóstico real** e não substitui avaliação médica.
