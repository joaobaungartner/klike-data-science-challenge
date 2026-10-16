# Parte 4 — Visão de Produto

Estas são **propostas futuras**, não funcionalidades implementadas ou resultados medidos no dataset. O desafio contém 500 campanhas simuladas com metadados, sem os vídeos originais.

## 1. Features adicionais dos vídeos

Eu priorizaria atributos extraíveis **antes da veiculação**, que permitam ações concretas e possam ser avaliados por custo/qualidade:

| Dimensão | Features potenciais | Utilidade de negócio |
| --- | --- | --- |
| Primeiros 3 segundos | início do hook, tempo até a proposta de valor, cortes e movimento iniciais | testar abertura mais objetiva |
| Visão computacional | número/tempo de rostos, produto em quadro, ritmo de cenas, enquadramento | contextualizar presença humana e montagem |
| OCR | texto por segundo, contraste, posição e duração de CTAs | analisar legibilidade e densidade de texto |
| Áudio/ASR | início da fala, velocidade, pausas, relação entre narração e legendas | testar variações de voz e legendagem |
| Identidade/mensagem | quando a marca aparece, clareza de benefício/CTA, consistência audiovisual | gerar sugestões alinhadas ao objetivo do anúncio |
| Semântica visual | representação multimodal do vídeo, tom e progressão narrativa | comparar criativos além dos metadados tabulares |

Poderia usar OCR, ASR e visão computacional para construir essas features. **Antes de treinar**, validaria a extração com amostra rotulada manualmente e avaliaria o custo por vídeo, privacidade de rostos/vozes e qualidade por plataforma. Só manteria novas variáveis se trouxessem ganho consistente em validação fora da amostra; embeddings não garantem melhoria.

## 2. Arquitetura de produção e integração ao Klike

O protótipo Streamlit é uma prova de conceito. A arquitetura abaixo **não foi implementada**:

1. **API com esquema estável:** requisições de campanha por FastAPI; autenticação, validação das dez features e versionamento de contratos. Uploads de vídeo seriam enviados para armazenamento de objetos com URLs temporárias.
2. **Pré-processamento assíncrono:** fila de mensagens + workers para OCR, ASR e visão computacional, com reprocessamento em falhas. Separar uploads pesados da inferência do Ridge.
3. **Serviço de inferência:** carregar pipeline Ridge e ordem de features versionados, manter métricas pós-veiculação fora da entrada e devolver predições por cenários.
4. **Serviço de evidências contextuais:** construir grupos por plataforma/categoria a partir de dados aprovados para treinamento; pré-calcular/cachear estatísticas e bootstrap para reduzir latência.
5. **Formato de resposta:** devolver ação, Δ previsto, intervalo observacional, tamanho amostral e status. Mostrar oportunidades exploratórias separadamente de recomendações elegíveis, sem afirmar causalidade.
6. **Logs/monitoramento:** latência, distribuição dos inputs, categorias desconhecidas, erros, drift, cobertura dos grupos, resultados futuros; controle de versões, testes de regressão, rollback e rollout gradual.
7. **Segurança:** acesso restrito, criptografia em trânsito e repouso, retenção limitada e tratamento seguro de voz/rosto conforme LGPD e necessidades de negócio.

**Fluxo de produto:** criar/analisar campanha → receber recomendações → comparar versões do criativo → executar testes A/B → associar resultados observados à campanha para monitoramento e futura melhoria do modelo. Resultados pós-veiculação não devem contaminar a predição pré-lançamento.

**Escalabilidade:** API sem estado, filas para vídeos, cache de evidências e armazenamento/versionamento dos modelos. O bootstrap atual é recalculado na interação; essa seria uma otimização prioritária para produção.

## 3. Se tivesse mais tempo

Em ordem de prioridade:

1. **Generalização e confiabilidade:** teste temporal em campanhas futuras, avaliação por plataforma/categoria, comparação com baseline, avaliação da estabilidade das diferenças previstas e quantificação da incerteza do próprio modelo.
2. **Validação causal:** experimentos A/B randomizados modificando um elemento por vez (hook/rosto/legendas), objetivos e métricas primárias definidos antes do teste. Sem isso, não chamar associações de *lift* causal.
3. **Robustez estatística:** tratar múltiplas comparações, suportes pequenos, diferenças contextuais, confundimento e possíveis interações entre plataforma e característica.
4. **Enriquecimento de dados:** features multimodais, objetivo da campanha e público-alvo, desde que disponíveis antes da veiculação e validadas sem leakage.
5. **Produto e UX:** comparação lado a lado, exportação de recomendações, validação com profissionais de marketing e medição da utilidade da sugestão.
6. **Operação:** CI/CD, acompanhamento de drift, otimização de tempo do bootstrap, monitoramento de integridade e retreinamento controlado.

## Conclusão

O motor atual usa o Ridge reduzido e associações exploratórias para ordenar sugestões; isso demonstra viabilidade de **apoio à decisão**, não prova de aumento de CTR, ROAS, receita ou score. A prioridade é validar o benefício real por experimentação antes de acrescentar complexidade ao modelo.
