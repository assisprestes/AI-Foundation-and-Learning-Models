# Quantum Commerce — Detecção de anomalias e fraudes

**Arquitetura híbrida de Machine Learning (Isolation Forest + Gradient Boosting) para detectar fraudes em transações e devoluções no varejo omnicanal.**

Trabalho Final da disciplina **Machine Learning Foundation and Classical Models** — FIAP MBA · AI Foundation and Learning Models · Turma 5AIER.

---

## Integrantes

| Nome | RM |
|---|---|
| Assis Prestes | RM377711 |
| Rafael Zaneti | RM377483 |
| Alecxander Ribeiro de Oliveira | RM378613 |
| Danilo Sanches | RM378307 |

---

## O problema

A Quantum Commerce (empresa fictícia do estudo de caso) é uma varejista omnicanal nativa digital, presente em 12 países e com mais de 5 milhões de SKUs. Nessa escala, regras fixas e revisão manual não acompanham o volume de pedidos nem a velocidade com que os golpes mudam — o chamado *muro da escalabilidade operacional*.

Escolhemos o desafio de **detecção de anomalias e fraudes**, que cobre três frentes:

- **Pagamentos:** cartão roubado, teste de cartões em série, tomada de conta (*account takeover*)
- **Devoluções:** usar e devolver, caixa vazia, reembolso sem devolução
- **Contas e promoções:** múltiplas contas para abusar de cupons e programas de fidelidade

## Por que uma arquitetura híbrida

O problema tem quatro características que nenhum modelo isolado resolve bem:

1. **Classes muito desbalanceadas** — fraude é menos de 1% dos pedidos; acurácia engana.
2. **Rótulos tardios e incompletos** — o chargeback chega semanas depois; muita fraude nunca é confirmada.
3. **Padrões que mudam** — golpes novos não têm histórico para um modelo supervisionado aprender.
4. **Decisão rápida e explicável** — o score precisa sair no checkout e ser justificável.

A solução combina:

| Componente | Tipo | Papel |
|---|---|---|
| **Isolation Forest** | Não supervisionado | Detecta comportamento atípico **sem rótulo** — cobre a fraude inédita |
| **Gradient Boosting** (XGBoost / HistGradientBoosting) | Supervisionado | Aprende os padrões de fraude **já confirmados** |
| **Stacking + regra de anomalia extrema** | Combinação | O score de anomalia vira feature do Gradient Boosting; anomalias acima do p99,5 vão no mínimo para revisão |

O score final é calibrado (regressão isotônica) e convertido em quatro ações por **limiares de custo**: aprovar, autenticar (3-D Secure), revisar ou bloquear.

---

## Prova de conceito (notebook)

O notebook `QuantumCommerce_Deteccao_Fraudes.ipynb` implementa a proposta de ponta a ponta:

| # | Etapa |
|---|---|
| 1 | Geração de base sintética de transações |
| 2 | Divisão temporal (treino / validação / teste) |
| 3 | Baseline — regressão logística |
| 4 | Isolation Forest treinado só com transações legítimas |
| 5 | Gradient Boosting supervisionado |
| 6 | Modelo híbrido |
| 7 | Calibração e limiares escolhidos por custo |
| 8 | Avaliação (PR-AUC e recall com 1% de falso positivo) |
| 9 | Decisão operacional nas quatro faixas |
| 10 | Explicabilidade (importância das features e motivos de cada decisão) |
| 11 | Simulação da API de scoring do checkout |
| 12 | Monitoramento de drift com PSI |

### Base de dados

Como não há dados reais da empresa, o notebook gera uma **base sintética** com os mesmos atributos da proposta:

- 151.200 transações em 180 dias, 12 países e 4 canais
- 0,72% de fraude rotulada, com 10% das fraudes propositalmente sem rótulo
- 3 tipos de fraude: **teste de cartões** e **tomada de conta** em todo o período, e **abuso de devolução**, um golpe novo que aparece **só no último mês** (ausente do treino)

| Conjunto | Período | Uso |
|---|---|---|
| Treino | dias 0–119 | ajuste dos modelos |
| Validação | dias 120–149 | calibração e limiares |
| Teste | dias 150–179 | avaliação, inclui o golpe novo |

### Resultados no mês de teste

Recall medido com no máximo **1% de falso positivo** sobre clientes legítimos:

| Modelo | PR-AUC | Fraude conhecida | Fraude nova |
|---|---|---|---|
| Regressão logística | 0,53 | 97% | 14% |
| Isolation Forest | 0,82 | 87% | 100% |
| Gradient Boosting | 0,74 | 99% | 66% |
| **Híbrido** | **0,86** | **98%** | **99%** |

**Leitura:** o Gradient Boosting é o melhor na fraude conhecida, mas deixa escapar um terço do golpe novo. O Isolation Forest enxerga o golpe novo sem rótulo, mas perde parte da fraude conhecida. O híbrido une os dois.

Outros resultados do protótipo:

- **Decisão:** 98% dos pedidos aprovados sem atrito; 99% das fraudes bloqueadas ou enviadas para revisão.
- **Explicabilidade:** o score de anomalia é a feature mais importante do híbrido, seguido da distância entre IP e endereço de entrega.
- **Latência:** cerca de 100 ms por pedido no protótipo em pandas, sem otimização.
- **Drift:** numa promoção simulada, o PSI do valor do pedido foi 0,49 (acima de 0,25), disparando o alerta de retreino.

> ⚠️ **Limitação:** a base é sintética e as fraudes se separam bem das transações legítimas, então os números são otimistas. Eles validam o **método**, não o desempenho esperado em produção.

---

## Como executar

### Google Colab

1. Abra o [Google Colab](https://colab.research.google.com/) e faça upload do arquivo `QuantumCommerce_Deteccao_Fraudes.ipynb`.
2. Execute todas as células (*Ambiente de execução → Executar tudo*).

### Localmente

```bash
git clone <url-deste-repositorio>
cd <pasta-do-repositorio>
pip install -r requirements.txt
jupyter notebook QuantumCommerce_Deteccao_Fraudes.ipynb
```

### Dependências

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn` (obrigatórias)
- `xgboost` e `shap` (opcionais)

O notebook detecta automaticamente se `xgboost` e `shap` estão instalados. Sem eles, usa o `HistGradientBoostingClassifier` do scikit-learn — mesma família de algoritmo — e uma explicação local simplificada. Os resultados acima foram gerados com o scikit-learn; com XGBoost os números podem variar um pouco.

A semente aleatória é fixa (`42`), então a execução é reprodutível.

---

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── QuantumCommerce_Deteccao_Fraudes.ipynb   # prova de conceito executada
└── docs/
    └── apresentacao.pdf                     # slides do trabalho (exportados)
```

---

## Próximos passos (roadmap da proposta)

| Período | Fase | Entregas |
|---|---|---|
| Meses 0–3 | MVP e modo sombra | piloto em 1 país, feature store inicial, baseline de métricas |
| Meses 3–6 | Produção e expansão | checkout e devoluções, teste A/B por país, rollout para os 12 países |
| Meses 6–12 | Aprendizado contínuo | monitoramento de drift, retreino automatizado, estudo de grafos (GNN) para redes de fraude |

---

## Referências

- [scikit-learn — IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [scikit-learn — HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (alternativa de base pública)

---

<sub>Projeto acadêmico. Quantum Commerce é uma empresa fictícia; todos os dados são sintéticos.</sub>
