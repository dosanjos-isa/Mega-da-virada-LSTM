# Configurações de experimentos

Este diretório centraliza os arquivos YAML de configuração dos experimentos, conforme definido na especificação (RC-003). Nenhuma constante mágica deve existir no código — todos os hiperparâmetros devem vir destes arquivos.

## Estrutura

```
configs/
├── README.md              # este arquivo
└── experiments/           # configurações individuais de experimentos
```

## O que cada YAML deve registrar

Conforme RC-003, cada execução deve gravar:

- Configuração utilizada (hiperparâmetros)
- Hash SHA-256 do dataset
- Versões das bibliotecas
- Sementes utilizadas
- Hash do commit

## Hiperparâmetros de referência (seção 5 da spec)

| Parâmetro | Valores / padrão |
|-----------|-----------------|
| Janela (W) | {10, 20, 40} — padrão: 20 |
| Peso α (regulares) | {0,01; 0,05; 0,10; 0,25} + controle 1,0 |
| Otimizador | Adam, `learning_rate = 1e-3` |
| ReduceLROnPlateau | fator 0,5; paciência 5 |
| Tamanho do lote | 64 |
| Épocas máximas | 100 |
| Parada antecipada | paciência 10 (perda de validação sem peso) |
| Validação | último 10% cronológico do treino |
| Dropout | 0,2 |
| `recurrent_dropout` | testado na grade |
| Sementes | mínimo 10 por configuração |
| Grade de busca | no máximo 20 configurações |

## Convenções

- Os YAMLs de experimentos individuais ficam em `experiments/`.
- Todas as configurações executadas devem ser registradas, incluindo as que não apresentaram bons resultados (RV-005).
- Alterações nos valores de referência acima exigem revisão explícita da spec.

## Referência

Consultar a especificação completa em [`spec/spec_lstm_mega_virada_v1.md`](../spec/spec_lstm_mega_virada_v1.md), seções 5 e 8.
