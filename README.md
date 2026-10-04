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

(a preencher)

## Avaliação dos modelos, governança e vieses

(a preencher)

## Aviso

Projeto acadêmico com dados 100% simulados. **Não é uma ferramenta de diagnóstico real** e não substitui avaliação médica.
