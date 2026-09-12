# PoC: Visao Neuromorfica em Smartphones

## 1. Sobre o experimento

Esta PoC investiga se e possivel executar um pipeline de visao computacional baseada em eventos em um smartphone Android convencional, usando uma Rede Neural de Impulsos (SNN) em hardware movel de prateleira.

A hipotese experimental e:

> Uma SNN alimentada por eventos de um sensor DVS pode ser executada em um smartphone convencional por meio de uma representacao temporal densa, um grafo ONNX estatico e os recursos CPU/GPU/NPU disponiveis no SoC.

O resultado deve ser interpretado como **emulacao de computacao neuromorfica em hardware convencional**. Esta PoC nao afirma que o smartphone possui um processador neuromorfico dedicado.

## 2. Motivacao

CNNs tradicionais processam frames completos em frequencia fixa, mesmo quando poucos pixels mudam. Sensores DVS produzem eventos assincronos no formato `(x, y, timestamp, polarity)`, reduzindo redundancia em cenas com movimento esparso.

A pesquisa mede o custo real de transformar esses eventos em uma representacao temporal e executar uma SNN em um SoC comercial. Os principais riscos sao:

- custo de leitura e preparacao dos eventos;
- exportacao do estado temporal LIF para um grafo ONNX;
- suporte dos runtimes moveis a operadores do modelo;
- latencia, temperatura e energia no smartphone;
- possivel custo da densificacao do fluxo de eventos.

## 3. Objetivos

### Objetivo geral

Construir e medir um pipeline de ponta a ponta:

```text
DVS/AEDAT -> eventos -> voxel grid -> SNN desenrolada -> ONNX -> Android
```

### Objetivos especificos

- Ler gravacoes reais do DVS128 Gesture.
- Gerar voxel grids temporais com polaridade e interpolacao temporal.
- Treinar uma SNN leve com neurônios LIF e gradiente surrogate.
- Exportar o modelo para ONNX e verificar paridade numerica.
- Executar a inferencia no Android com ONNX Runtime Mobile.
- Comparar CPU, GPU e NNAPI quando o modelo estiver integrado ao aparelho.
- Medir latencia, throughput, memoria, temperatura e consumo.
- Comparar a SNN com uma baseline CNN densa sob a mesma tarefa.

## 4. Arquitetura planejada

```text
[Arquivo AEDAT / sensor DVS USB-OTG]
              |
              v
[Leitura de eventos]
  x, y, timestamp_us, polarity
              |
              v
[Pre-processamento nativo C++/NDK]
  janelas temporais, voxel grid/time-surface
              |
              v
[Ponte JNI / DirectByteBuffer]
              |
              v
[ONNX Runtime Mobile]
  CPU ARM -> GPU -> NNAPI, conforme suporte
              |
              v
[Aplicacao Android]
  predicao, latencia, temperatura, corrente
```

A implementacao atual esta na fase de integracao mobile. O pipeline host/Python ja foi validado com dataset real, exportacao ONNX e paridade numerica. A etapa atual e a prova de conceito Android com ONNX Runtime Mobile, usando o modelo exportado `host/artifacts/snn_gesture_trained.onnx`.

### Estado atual da fase Android

- estrutura do projeto Android configurada em `android/Spike4MobileApp`;
- ambiente local preparado com JDK 17, Android SDK (API 34) e Gradle 8.7;
- compatibilidade de Gradle e caminhos com caracteres especiais configurada (`android.overridePathCheck=true`);
- integracao do ONNX Runtime Mobile para Android (`ai.onnxruntime:onnxruntime-android:1.19.2`);
- task automatica no Gradle (`copyOnnxModel`) para sincronizar o artefato `host/artifacts/snn_gesture_trained.onnx` como asset da aplicacao;
- `MainActivity.kt` implementada com carregamento do modelo a partir de assets, construcao de tensor `[1, 8, 2, 32, 32]`, execucao de sessao ONNX e exibicao da predicao/logits na interface;
- build Android validado com sucesso (`assembleDebug`);
- aplicativo instalado (`:app:installDebug`) e executado com sucesso no emulador Android (`Pixel_6_Pro_API_UpsideDownCake` / `emulator-5554`), sem falhas em tempo de execucao;
- `.gitignore` atualizado para ignorar artefatos de build e temporarios do Android (`.gradle`, `**/build`, `local.properties`, `.cxx`, `.externalNativeBuild`, etc.).

## 5. Dataset

Foi usado o dataset publico DVS Gesture disponibilizado no Kaggle:

```text
xingfenyizhen/dvsgesture128
```

O download foi feito com o cliente oficial `kagglehub` e esta em:

```text
data/dvsgesture128
```

O dataset contem arquivos AEDAT 3.1, anotacoes por janela de gesto, 11 classes e sensor de resolucao `128x128`. Os dados nao devem ser versionados.

O leitor implementado trata o formato binario observado no dataset, incluindo preambulo textual e pacotes little-endian.

## 6. Estrutura do repositorio

```text
README.md                         este documento
host/aedat3.py                    leitor AEDAT 3.1 e anotacoes
host/event_voxel.py               gerador de voxel grid
host/aedat_smoke_test.py          teste do leitor com dados reais
host/voxel_smoke_test.py          teste do voxel grid sintetico
host/parity_smoke_test.py         paridade PyTorch/ONNX com modelo smoke
host/real_pipeline_smoke_test.py  AEDAT real -> voxel grid -> ONNX
host/train_snn.py                 treinamento da SNN
host/evaluate_snn.py              avaliacao em trials separados
host/export_trained_snn.py        exportacao do checkpoint treinado para ONNX
host/requirements.txt              dependencias Python
host/artifacts/                   modelos, checkpoints e cache local
android/Spike4MobileApp/          projeto Android Kotlin com ONNX Runtime Mobile
  app/src/main/java/...           MainActivity com inferencia ONNX
  app/src/main/assets/            modelo ONNX copiado durante o build
  app/build.gradle.kts            configuracoes de build e task copyOnnxModel
data/                             dataset local, nao versionar
```

## 7. Preparacao do ambiente

O ambiente de referencia usa Python 3.11 no Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r host\requirements.txt
.\.venv\Scripts\python.exe -m pip install kagglehub
```

Baixar o dataset publico:

```powershell
.\.venv\Scripts\python.exe -c "import kagglehub; kagglehub.dataset_download('xingfenyizhen/dvsgesture128', output_dir='data/dvsgesture128')"
```

## 8. Execucao passo a passo

Execute os comandos a partir da raiz do projeto.

### 8.1 Validar o voxel grid

```powershell
.\.venv\Scripts\python.exe host\voxel_smoke_test.py
```

Valida formato `[bins, 2, height, width]`, canais de polaridade, interpolacao temporal e conservacao da massa dos eventos.

### 8.2 Validar o leitor AEDAT

```powershell
.\.venv\Scripts\python.exe host\aedat_smoke_test.py
```

Valida coordenadas, polaridade, timestamps e recorte de uma janela real anotada.

### 8.3 Validar PyTorch contra ONNX

```powershell
.\.venv\Scripts\python.exe host\parity_smoke_test.py
```

Gera `host/artifacts/snn_smoke.onnx` com entrada sintetica deterministica. Este modelo tem pesos aleatorios e serve somente para validar exportacao e runtime.

### 8.4 Validar o pipeline com eventos reais

```powershell
.\.venv\Scripts\python.exe host\real_pipeline_smoke_test.py
```

Este teste usa uma janela real, reduz DVS128 para `32x32` para coincidir com o modelo smoke e executa ONNX Runtime.

### 8.5 Treinar a SNN

Rodada curta:

```powershell
.\.venv\Scripts\python.exe host\train_snn.py --epochs 1 --max-trials 1 --max-gestures 4
```

Rodada incremental:

```powershell
.\.venv\Scripts\python.exe host\train_snn.py --epochs 1 --max-trials 2 --max-gestures 16
```

O checkpoint e salvo em `host/artifacts/snn_gesture_checkpoint.pt`.

O treinamento usa cache por trial em `host/artifacts/sample_cache`. As mensagens indicam o estado:

```text
CACHE_BUILD  trial ainda nao processado
CACHE_SAVED  trial convertido e salvo
CACHE_HIT    trial reutilizado
```

### 8.6 Avaliar a SNN

```powershell
.\.venv\Scripts\python.exe host\evaluate_snn.py --max-trials 1 --max-gestures 16
```

A avaliacao usa `trials_to_test.txt`, separado dos trials de treino.

### 8.7 Exportar a SNN treinada para ONNX

```powershell
.\.venv\Scripts\python.exe host\export_trained_snn.py
```

O script carrega o checkpoint, gera `host/artifacts/snn_gesture_trained.onnx` com entrada `[batch, tempo, canais, altura, largura]` e compara ONNX Runtime com PyTorch em uma amostra real cacheada.

### 8.8 Compilar e instalar o app Android

A partir do diretorio `android/Spike4MobileApp`:

```powershell
Set-Location android\Spike4MobileApp
$env:ANDROID_HOME = 'C:...\AppData\Local\Android\Sdk'
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-17'

# Compilar APK de debug (copia o modelo ONNX automaticamente)
.\gradlew.bat assembleDebug

# Instalar no emulador ou dispositivo conectado
.\gradlew.bat installDebug

# Abrir o aplicativo via adb
& "$env:ANDROID_HOME\platform-tools\adb.exe" shell am start -n com.spike4mobile.app/com.spike4mobile.app.MainActivity
```

## 9. Resultados obtidos

Os testes foram executados em um Acer Nitro 5 com 16 GB de RAM e GTX 1650. O ambiente Python atual usa `torch+cpu`, portanto a GTX 1650 ainda nao participa do treinamento. A validacao mobile foi feita em emulador Android (API 34 / Pixel 6 Pro).

| Teste                    | Resultado                                                                  |
| ------------------------ | -------------------------------------------------------------------------- |
| Voxel grid sintetico     | `VOXEL_OK shape=(4, 2, 4, 4) total=3.0`                                    |
| Leitura AEDAT real       | `AEDAT_OK events=1000 first_gesture_events=1000 labels=12 sensor=128x128`  |
| Paridade PyTorch/ONNX    | `PARITY_OK max_abs_error=0.00000000`                                       |
| Pipeline real            | `PIPELINE_OK events=132219 input=(8, 1, 2, 32, 32) output=(1, 3)`          |
| Treino minimo            | 4 amostras, 1 epoca, 24,0 s, checkpoint criado                             |
| Avaliacao minima         | 4 amostras, accuracy `0.000`                                               |
| Treino incremental       | 16 amostras, loss `2.4412`, accuracy `0.062`, 72,6 s na primeira rodada    |
| Avaliacao incremental    | 12 amostras, accuracy `0.083`                                              |
| Cache reutilizado        | mesma rodada caiu de 28,2 s para 6,8 s                                     |
| Cache por trial          | 2 trials reutilizados, 16 amostras, treino em 3,1 s                        |
| Escala para 5 trials     | 4 trials cacheados; o quinto foi interrompido durante leitura AEDAT        |
| Leitura streaming        | 11 janelas nao vazias em `user01_natural`, treino cacheado em 2,9 s        |
| Treino em 5 trials       | 50 gestos, 1 epoca, loss `2.4374`, accuracy `0.080`, 59,8 s                |
| Avaliacao em 3 trials    | 30 gestos, accuracy `0.067`, checkpoint de 5 trials                        |
| Treino representativo    | 100 gestos, 5 epocas, loss `2.4244`, accuracy `0.080`, 17,3 s              |
| Avaliacao representativa | 30 gestos, accuracy `0.067`, checkpoint de 10 trials                       |
| Exportacao treinada      | entrada `(1, 8, 2, 32, 32)`, saida `(1, 11)`, erro maximo `0.00000006`     |
| Compilacao Android       | `BUILD SUCCESSFUL` com Gradle 8.7, JDK 17 e ONNX Runtime Android 1.19.2    |
| Execucao Android         | APK instalado e aberto no emulador (`emulator-5554`), sem erros de runtime |

As acuracias acima nao sao resultados cientificos. O treinamento ainda usa poucas amostras e uma arquitetura experimental. Com 11 classes, a referencia aleatoria e aproximadamente `9,1%`.

## 10. Plano de testes

1. Repetir o treino incremental tres vezes com a mesma seed e registrar variacao de tempo, loss e acuracia.
2. Registrar matriz de confusao e acuracia por classe.
3. Repetir paridade PyTorch/ONNX em mais amostras reais.
4. [CONCLUIDO] Criar a aplicacao Android com ONNX Runtime Mobile, sincronizacao de modelo e execucao basica.
5. Medir inferencia no smartphone com CPU, GPU (NNAPI / XNNPACK) quando suportados.
6. Comparar contra uma CNN densa sob o mesmo dataset.
7. Medir energia com telemetria Android e, idealmente, medidor externo USB-C.

Metricas finais:

- latencia de leitura e pre-processamento;
- latencia de inferencia;
- janelas e eventos processados por segundo;
- memoria;
- acuracia total e por classe;
- corrente e potencia;
- temperatura antes, durante e depois do teste.

## 11. Hardware e limites

O Acer Nitro 5 continua suficiente para a PoC atual. O gargalo encontrado foi a leitura completa de trials AEDAT grandes. A leitura agora percorre cada trial uma vez e retém somente eventos dentro das janelas anotadas; combinada ao cache por trial, ela reduz o uso de memoria e permite retomar o processamento. Nao houve erro de memoria.

- Acer atual: suficiente para smoke tests, cache, treino incremental e exportacao.
- CUDA: recomendavel a partir de treinos com pelo menos 10 trials ou varias epocas.
- PC mais potente: somente se uma rodada completa ficar operacionalmente impraticavel ou se forem necessarias muitas buscas de hiperparametros.
- Smartphone: necessario para medir latencia, temperatura, consumo e backends moveis.

A GTX 1650 pode ser aproveitada no futuro instalando uma distribuicao PyTorch com CUDA compativel. Isso e uma otimizacao de tempo, nao um requisito para concluir a PoC basica.

## 12. Cuidados de interpretacao

- Executar no Android nao equivale a usar um chip neuromorfico dedicado.
- `DirectByteBuffer` evita uma copia explicita entre JNI e JVM, mas zero-copy ponta a ponta precisa ser medido.
- `BATTERY_PROPERTY_CURRENT_NOW` e temperatura Android sao medidas auxiliares e podem variar por fabricante.
- Para energia publicavel, preferir medidor externo.
- O modelo smoke e o checkpoint de poucas amostras nao devem ser usados para afirmar acuracia da pesquisa.
- O modelo final deve ser exportado e validado antes de qualquer comparacao de backends.
