# Video Dublador

Pipeline completo de dublagem automática de vídeos: **recebe um vídeo, transcreve,
traduz, clona a voz de cada ator, gera a nova faixa de áudio, sincroniza os lábios
e devolve o vídeo dublado** no idioma escolhido.

```
vídeo ──► extrai áudio ──► transcreve (Whisper)
      ──► analisa as vozes (diarização + escolhe clipes de referência)
      ──► traduz
      ──► clona cada voz e sintetiza (F5-TTS)
      ──► encaixa cada fala na sua janela de tempo original
      ──► remonta a linha do tempo do áudio
      ──► re-anima os lábios (Wav2Lip) e faz o mux
      ──► vídeo dublado
```

---

## 1. Instalação

O projeto é **autocontido e portátil**: o Python, o ffmpeg e todos os modelos
ficam dentro da própria pasta. Nada é instalado no sistema (não precisa de
administrador).

```powershell
cd video-dublador
.\run.ps1 check          # diagnóstico do ambiente
```

Se for construir o ambiente do zero em outra máquina:

```powershell
python tools\bootstrap.py
```

### O que já está provisionado nesta máquina

| Componente | Versão | Local |
|---|---|---|
| Python | 3.11.9 (embeddable) | `.python\` |
| pip | 26.2.1 | `.python\Lib\site-packages` |
| ffmpeg / ffprobe | 9.0.1 essentials | `.tools\ffmpeg\bin\` |
| PyTorch | 2.14.0+cpu | site-packages |
| faster-whisper | 1.2.1 (CTranslate2) | site-packages |
| F5-TTS | 1.1.22 | site-packages |
| deep-translator | 1.11.4 | site-packages |

---

## 2. Interface gráfica (jeito mais fácil)

Não é preciso digitar comandos. **Dê dois cliques em `ABRIR-INTERFACE.cmd`** —
o navegador abre sozinho em `http://127.0.0.1:8760/` e você faz tudo por ali.

```
ABRIR-INTERFACE.cmd            ← dois cliques, pronto
```

A janela preta que abre **precisa ficar aberta** enquanto você usa a interface;
fechá-la (ou `Ctrl+C`) encerra o servidor local.

O que a interface faz:

| Área | Para quê |
|---|---|
| **1. Vídeo** | Escolhe um arquivo de `input/`, ou **arrasta e solta** um vídeo na área pontilhada para enviá-lo (até 4 GB) |
| **2. Idiomas** | Idioma de destino e idioma original (ou detecção automática) |
| **3. Qualidade** | Presets **Rápido / Equilibrado / Qualidade** — um clique |
| **Avançado** | Modelo do Whisper, passos do F5-TTS, modelo de voz, tradutor, número de locutores, trilha original, “parar após” e as opções de lip sync |
| **Progresso** | Barra de progresso, etapa atual (1/7 … 7/7), tempo decorrido e botão **Cancelar** |
| **Registro ao vivo** | O log completo do pipeline em tempo real |
| **Saídas geradas** | Lista dos vídeos dublados com link de **download** |
| **Histórico** | Os últimos 12 trabalhos, com estado e duração |

Detalhes úteis:

- As configurações escolhidas ficam salvas no navegador (voltam na próxima visita).
- **Cancelar é seguro**: cada etapa é gravada em disco. Clicar em “Dublar” de novo
  continua de onde parou, sem refazer o que já terminou.
- O botão **Cancelar** encerra o processo do pipeline de verdade (não deixa
  trabalho órfão rodando em segundo plano).
- **“Parar após”** é a saída de emergência para esta máquina lenta: pare na
  *Tradução* para conferir o texto antes de gastar horas em síntese de voz. Depois
  rode de novo escolhendo “Concluir tudo” — o que já foi feito é reaproveitado.
- Os selos no topo mostram o estado do ambiente (ffmpeg, torch, F5-TTS, Wav2Lip,
  espaço em disco). Passe o mouse para ver os detalhes.

> **Segurança:** o servidor escuta **somente em 127.0.0.1** (ninguém da rede
> alcança) e toda requisição que altera algo exige um token aleatório gerado a
> cada execução, embutido na página. Assim, uma aba aberta em outro site não
> consegue acionar a API local.
>
> A interface **não abre uma porta para a internet** e não substitui nada: é só
> um painel local que chama o mesmo `python -m dublador` que você usaria no
> console.

Opções de linha de comando, se precisar:

```powershell
.\run.ps1 gui                      # porta padrão 8760, abre o navegador
.\run.ps1 gui --port 8900          # outra porta
.\run.ps1 gui --no-browser         # não abre o navegador
```

---

## 2b. Uso pela linha de comando

```powershell
# lista os idiomas suportados
.\run.ps1 languages

# dubla um vídeo para português
.\run.ps1 input\filme.mp4 --target pt

# do inglês para o espanhol, voz e legendas
.\run.ps1 filme.mp4 --target es --source en

# prévia rápida (Whisper small, 16 passos, sem lip sync)
.\run.ps1 filme.mp4 --target pt --fast

# melhor qualidade (Whisper large-v3 + lip sync)
.\run.ps1 filme.mp4 --target pt --quality

# só transcrever (teste barato, não baixa o modelo do F5-TTS)
.\run.ps1 filme.mp4 --transcribe-only

# manter trilha sonora/música original por baixo da dublagem
.\run.ps1 filme.mp4 --target pt --background duck
```

### Opções principais

| Opção | Descrição |
|---|---|
| `-t, --target` | **idioma de destino** (obrigatório), ex. `pt`, `en`, `es` |
| `-s, --source` | idioma de origem (padrão: detectar automaticamente) |
| `-o, --output` | caminho do vídeo de saída |
| `--whisper-model` | `tiny`…`large-v3` (padrão `small`) |
| `--f5-model` | `F5TTS_v1_Base` (padrão) ou `F5TTS_v1_Small` |
| `--nfe-step` | passos de denoising: 32 = qualidade, 16 = ~2× mais rápido |
| `--no-fix-duration` | deixa o F5-TTS escolher o tamanho em vez de preencher a janela |
| `--no-lipsync` | pula o Wav2Lip |
| `--background` | `none` (padrão) / `duck` / `separate` |
| `--min-speakers` / `--max-speakers` | limites de locutores |
| `--no-resume` | ignora cache e refaz tudo |
| `-v, --verbose` | log detalhado + traceback completo |

---

## 2c. Detecção automática de hardware

**Você não precisa dizer ao programa que máquina é essa.** Na primeira execução
ele detecta CPU, GPU, memória e disco, classifica o conjunto e escolhe sozinho
dispositivo, threads, modelos e passos de síntese. Depois disso a detecção fica
em cache por 6 horas.

Para ver o que ele encontrou:

```powershell
.\run.ps1 hw                  # relatório completo + ajustes escolhidos
.\run.ps1 hw --benchmark      # mede a máquina de verdade (estimativa precisa)
.\run.ps1 hw --json           # saída para outros programas
.\run.ps1 hw --refresh        # ignora o cache e redetecta
```

Exemplo real, nesta máquina:

```
HARDWARE DETECTADO
  CPU          Intel(R) Core(TM) i3-4130 CPU @ 3.40GHz
               2 nucleos fisicos / 4 threads  (AMD64)
  Inferencia   CTranslate2 suporta: float32, int16, int8, int8_float32
  Memoria      7.9 GB total, 1.9 GB livre (high)
  GPU          Intel(R) HD Graphics 4400 - nao utilizavel (integrada, sem CUDA)
  Disco        11.6 GB livres de 118.7 GB
  Sistema      Windows 10 | Python 3.11.9 | torch 2.14.0+cpu

CLASSIFICACAO  CPU modesta  (cpu-modest)

AJUSTES ESCOLHIDOS AUTOMATICAMENTE
  dispositivo        cpu
  threads            2
  Whisper            small / int8
  modelo de voz      F5TTS_v1_Base  (16 passos)
  lip sync           sim (resize 2)
  velocidade prevista ~57x o tempo real -> 1 min de video em ~57 min

OBSERVACOES
  - 4 threads lógicos / 2 núcleos físicos: limitando a 2 threads
    (hyper-threading atrapalha kernels densos).
  - RAM livre baixa (1.9 GB).
```

### O que é detectado

| Item | Como | Observação |
|---|---|---|
| Modelo da CPU | registro do Windows / `/proc/cpuinfo` | sem dependências |
| **Núcleos físicos** | `psutil`, com dedução por hyper-threading | o número que importa: threads lógicas *pioram* kernels densos |
| Tipos de inferência | `ctranslate2.get_supported_compute_types("cpu")` | responde na prática "este CPU faz int8 rápido?" (AVX2) |
| Memória | `psutil`, senão `ctypes.GlobalMemoryStatusEx` | total **e livre** (pressão de memória muda a recomendação) |
| GPU utilizável | `torch.cuda` / `torch.xpu` / `torch.backends.mps` | NVIDIA, AMD (ROCm), Intel (XPU), Apple |
| GPU presente mas inútil | registro do Windows / `/sys/class/drm` | ex.: Intel HD integrada — aparece, mas marcada como não utilizável |
| Disco | `shutil.disk_usage` | avisa se os ~2 GB de modelos não couberem |
| Velocidade | benchmark numpy/OpenBLAS | ver abaixo |

### Classificação e ajustes

| Tier | Condição | Whisper | Passos | Voz | Lip sync |
|---|---|---|---|---|---|
| `gpu-strong` | GPU ≥ 10 GB VRAM | large-v3 | 32 | Base | sim |
| `gpu-modest` | GPU ≥ 6 GB | medium | 32 | Base | sim |
| `gpu-small` | GPU < 6 GB | small | 24 | Base | sim |
| `gpu-apple` | Apple Silicon (MPS) | medium | 24 | Base | sim |
| `cpu-strong` | ≥ 8 núcleos e ≥ 16 GB | medium | 24 | Base | sim |
| `cpu-modest` | ≥ 2 núcleos e ≥ 6 GB | small | 16 | Base | sim |
| `cpu-minimal` | abaixo disso | base | 12 | Small | não |

Além disso o programa ajusta:

- **threads = núcleos físicos** (não lógicos), avisando quando há hyper-threading;
- **tipo de computação** do Whisper: `float16` na GPU, `int8` na CPU — e se o
  CPU não suportar o escolhido, cai para um que ele suporte;
- **reduz passos** quando a RAM livre está crítica;
- **avisa** se o disco não comporta os modelos.

### Como a velocidade é estimada

Em vez de chutar, o programa se calibra:

1. Roda um **benchmark de matmul float32** (numpy/OpenBLAS, melhor de 5
   amostras — a média oscila demais e distorceria toda previsão).
2. Compara com a **máquina de referência** deste projeto — o i3-4130 onde todas
   as medições do README foram feitas: **108,2 GFLOPS** e **~55× o tempo real**
   com F5-TTS Base a 16 passos.
3. Extrapola por núcleos físicos e, se houver GPU, aplica o ganho esperado.

Resultado nesta máquina: **57× previsto contra 55× medido** — cerca de 4% de
erro. Sem rodar `--benchmark` ele usa só o número de núcleos e também acerta
(55×), marcando a estimativa como não calibrada.

### Estimando quanto tempo vai levar

```powershell
.\run.ps1 estimate --duration 1h30m                 # nesta máquina
.\run.ps1 estimate --duration 1h30m --no-lipsync
.\run.ps1 estimate --duration 5m --speech-ratio 0.6

# "e se eu tivesse outra máquina?"
.\run.ps1 estimate --duration 1h30m --cores 6 --gflops 400 --vram 8 --gpu-name "RTX 5060"
```

Ele reporta **etapa por etapa**, porque cada uma escala de um jeito diferente:

| Etapa | Escala com | Observação |
|---|---|---|
| Extração de áudio | duração do vídeo | barato, ligado a I/O |
| Transcrição | duração do vídeo ÷ CPU/GPU | GPU acelera muito |
| Análise de vozes | **quantidade de fala** | CPU/librosa |
| Tradução | **número de falas** | limitada por API web, não por hardware |
| Síntese F5-TTS | **quantidade de fala** | normalmente domina |
| **Lip sync** | **duração total do vídeo × fps** | é por quadro — domina em filmes |
| Codificação | duração + se re-codifica | NVENC ajuda muito |

A fração de fala (`--speech-ratio`, padrão 0,5) importa muito: a síntese cobra
por **fala**, o lip sync cobra pelo **vídeo inteiro**.

Um ponto de honestidade embutido: em vídeos acima de 90 s o pipeline **pula o
lip sync sozinho**, então o total mostrado já reflete esse comportamento, e o
custo de forçá-lo aparece separado em "alternativas".

### Como sobrepor

Qualquer valor explícito vence a detecção:

```powershell
.\run.ps1 filme.mp4 --target pt --whisper-model large-v3 --nfe-step 32 --device cuda
```

Na interface gráfica existe o preset **Automático** (padrão) e o cartão
“Hardware detectado automaticamente”, que mostra as especificações, o tier, a
previsão de tempo e o que será usado. Os presets Rápido/Equilibrado/Qualidade
fixam valores concretos.

---

## 3. Como funciona cada etapa

### 3.1 Extração de áudio
Três faixas são extraídas do vídeo original:
- **16 kHz mono** — usada pelo Whisper e pela análise de voz;
- **24 kHz mono** — fonte para os clipes de referência da clonagem;
- **24 kHz** — trilha original, reaproveitada como fundo (`--background duck`).

### 3.2 Transcrição (`transcribe.py`)
`faster-whisper` (CTranslate2) com *word timestamps* e filtro VAD. Falas muito
longas (>12 s) são divididas em fronteiras de palavra, porque a qualidade da
dublagem cai em trechos longos.

### 3.3 Análise de voz (`speakers.py`)
Duas etapas:

1. **Diarização** — cada fala recebe um *embedding* de locutor e os embeddings
   são agrupados. Dois backends:
   - `speechbrain` (ECAPA-TDNN, 192-d) — mais preciso, se instalado;
   - `mfcc` — *fallback* sem download, com MFCC + deltas + F0 + forma espectral.

   O **número de locutores é escolhido pelos dados**, não por um limiar fixo:
   testa-se cada `k` possível, pontua-se cada partição com o *silhouette
   coefficient* (invariante à escala, então funciona igual nos dois backends) e
   fica-se com o melhor. Se nem a melhor divisão for convincente, assume-se
   **um único locutor** — resposta segura: gera uma voz clonada em vez de
   fragmentar um ator em três. Use `--speaker-threshold` para forçar um corte
   fixo de similaridade de cosseno.

   > **Limitação medida neste host:** o backend `mfcc` só separa com confiança
   > vozes bem distintas. Num teste com duas vozes sintéticas parecidas, a
   > similaridade de cosseno ficou entre 0,94 e 0,97 para *todos* os pares —
   > indistinguível. Ainda assim o agrupamento automático acertou a contagem
   > (2 locutores). Para diarização séria instale o `speechbrain`:
   > `.python\python.exe -m pip install speechbrain`

2. **Clipes de referência** — para cada locutor escolhe-se o trecho mais limpo
   (duração próxima do ideal, fala densa, alta confiança do ASR, sem silêncio) e
   ele é cortado do áudio original. **Esse par (áudio + texto) é a "impressão
   digital" da voz** que o F5-TTS vai clonar.

### 3.4 Tradução (`translate.py`)
Uma tradução **por segmento** (nunca em lote concatenado), porque a dublagem
exige correspondência 1:1 entre a fala original e a traduzida — um lote
mesclado destruiria o sincronismo. Tem cache em disco, *retries* com backoff e
cadeia de fallback (`google` → `mymemory` → `libre`).

> **Medido nesta rede:** o backend `google` do `deep-translator` **falha**
> (`TooManyRequests`) — o endpoint público do Google recusa as requisições
> desta rede. O fallback para o **MyMemory funciona** e é usado automaticamente:
> *"Good evening. The system is online."* → *"Boa noite. O sistema está online."*
> A implementação fala com a API REST do MyMemory diretamente via `urllib`
> (em vez de depender do `deep-translator`), porque aquele serviço exige códigos
> com locale (`pt-BR`, `en-GB`) e não apenas o ISO de duas letras. O primeiro
> erro de cada backend é reportado **uma vez** como aviso, para você saber qual
> está sendo usado de fato. O cache em `work/<job>/translation_cache.json`
> evita repetir chamadas de rede.

### 3.5 Clonagem e síntese (`tts.py`)
F5-TTS é um modelo de *flow matching* que clona uma voz a partir de um clipe
curto + sua transcrição, **sem fine-tuning por locutor**. Três recursos são
explorados:

1. **Clonagem** — cada locutor detectado vira um par de referência, então cada
   personagem mantém sua própria voz.
2. **`fix_duration`** — o F5-TTS é instruído a gerar uma fala que ocupe
   exatamente a duração da fala original. **É isso que mantém a dublagem no
   tempo da imagem sem artefatos de esticamento.**

   > ⚠️ **Armadilha do upstream (importante):** o parâmetro `fix_duration` do
   > F5-TTS é a duração **total, incluindo o clipe de referência** — o código
   > interno faz `target_total = fix_duration - ref_sec`. Passar só a duração da
   > janela (o que é natural) gera **áudio vazio** sempre que a referência for
   > mais longa que a janela. Como neste projeto as referências têm 3–6 s e as
   > falas podem ter 1,5 s, isso aconteceria quase sempre. O `dublador/tts.py`
   > soma a duração da referência antes de chamar o modelo **e** valida o
   > resultado: se o áudio sair absurdo para o tamanho pedido, ele repete a
   > síntese sem `fix_duration` e deixa o `fit_to_slot` cuidar do tempo.
3. **`speed`** — alavanca secundária quando `fix_duration` sozinho não basta.

### 3.6 Ajuste de tempo (`media.fit_to_slot`)
Se mesmo assim a fala gerada não couber, ela é esticada com **rubberband**
(preservando o tom) até o limite `--max-stretch` (padrão 1,40×); passando disso
é cortada. Falas mais curtas que a janela são apenas preenchidas com silêncio,
para a linha seguinte não sair do lugar.

### 3.7 Remontagem da linha do tempo (`assemble.py`)
Feita em **numpy**, não com um `filter_complex` gigante do ffmpeg: cada clipe é
somado na posição exata (`segment.start`) de um buffer do tamanho do vídeo.
Assim é exato ao sample, rápido, e não esbarra em limite de tamanho de linha de
comando em vídeos com centenas de falas.

### 3.8 Lip sync (`lipsync.py`)
Wav2Lip (rede + detector de face S3FD) roda em CPU. Se não houver face no vídeo
ou o backend não estiver instalado, o pipeline **degrada com elegância**: apenas
faz o mux do áudio dublado sobre a imagem original e avisa no log.

---

## 4. Desempenho nesta máquina (importante)

Hardware detectado:

| | |
|---|---|
| CPU | Intel Core i3-4130 @ 3,40 GHz — **2 núcleos físicos** |
| RAM | 7,9 GB total (**~1,3 GB livres**) |
| GPU | **nenhuma** (Intel HD Graphics 4400, sem CUDA) |
| Disco livre | ~12 GB |

Consequências práticas — **números medidos nesta máquina, não estimados**:

- **Tudo roda em CPU.** Não há CUDA.
- Um teste real de fumaça do F5-TTS gerou **6,47 s de fala em 1023 s** com
  `--nfe-step 32` (**~158× o tempo real**), medido com o CPU ainda disputado por
  outro processo. Com a máquina livre e `--nfe-step 16`, a medição caiu para
  **~56–64× o tempo real** (3,3 s de fala em ~195 s). Use esses números como
  referência: **~60× com 16 passos, ~150× com 32 passos**.
- O carregamento do modelo leva ~175 s na primeira vez e ~13 s depois (cache de disco).
- O **Wav2Lip custa ~1 s por frame** (a detecção de face domina): um clipe de
  1 minuto a 25 fps leva cerca de **25 minutos**.
- Na prática: **um vídeo de 1 minuto leva de 1 a 2 horas** para ser dublado
  nesta máquina. Vídeos longos são inviáveis sem GPU.

O pipeline é **retomável**: cada etapa é gravada em `work/<job>/`. Se você
interromper com `Ctrl-C`, rode o mesmo comando de novo e ele continua de onde
parou. Use `--no-resume` para forçar o recálculo.

Para acelerar:

| Ação | Efeito |
|---|---|
| `--nfe-step 16` | ~2× mais rápido, leve perda de qualidade |
| `--f5-model F5TTS_v1_Small` | modelo menor, mais rápido |
| `--no-lipsync` | pula a etapa mais lenta (~1 s/frame) |
| `--whisper-model base` | transcrição mais rápida |
| `--lipsync-resize 2` | Wav2Lip ~4× mais rápido, resultado mais suave |

O lip sync é **pulado automaticamente** acima de 90 s de vídeo (a menos que você
use `--lipsync-force`), justamente por causa desse custo.

---

## 4b. Verificação de ponta a ponta (executada de fato)

Não é teoria: o pipeline foi rodado neste host, do início ao fim, sobre um vídeo
de teste com **duas vozes distintas falando inglês**, dublado para português.

```
$ .\run.ps1 test_speech.mp4 --target pt --nfe-step 16 --no-lipsync

  STAGE 2  transcreveu 3 segmentos, idioma detectado en (100%)
  STAGE 3  detectou 2 locutores, extraiu 2 clipes de referência (5,1 s e 2,7 s)
  STAGE 4  traduziu en -> pt (fallback MyMemory; o Google recusou nesta rede)
  STAGE 5  7,0 s de fala gerada em 6m27s, fator de tempo real 54,9x
  STAGE 6  faixa dublada montada: 3 falas encaixadas, 11,70 s
  STAGE 7  mux concluído -> output\test_speech_pt_dubbed.mp4
  total    8m02s
```

Validação independente com `tools\verify_output.py` (extrai o áudio do vídeo
final e transcreve de volta):

```
  peak      : -4.0 dBFS          rms: -22.2 dBFS
  voiced    : 32,3% dos quadros acima do piso de ruído   -> áudio real, não silêncio
  transcrito: [0.62-2.98] Boa noite, o sistema esta online.
              [5.24-6.96] Demorou mais duquac promado.
              [8.84-9.74] Esta tudo pronto.
  sobreposição de palavras com a tradução gerada: 71%   -> PASSOU
```

Os tempos dos três trechos dublados batem com as janelas originais — ou seja, o
sincronismo com a imagem foi preservado. O `Wav2Lip` foi verificado à parte,
num vídeo com rosto real, e roda de ponta a ponta.

Ferramentas de verificação incluídas:

| Comando | O que faz |
|---|---|
| `tools\test_f5_smoke.py` | valida a síntese F5-TTS isoladamente e mede o custo real |
| `tools\verify_output.py` | confere se o vídeo final tem fala audível e no idioma certo |
| `tools\diag.py` | mostra a similaridade entre vozes e se a diarização é confiável |
| `tools\make_test_video.py` | gera um vídeo de teste com diálogo multi-locutor |
| `tools\test_gui.py` | testa a interface web de ponta a ponta (18 verificações) |

---

## 5. Saídas

```
output/<nome>_<idioma>_dubbed.mp4   ← vídeo dublado
work/<job>/
  ├── transcript.json               ← transcrição com timestamps por palavra
  ├── transcript_voices.json        ← locutores e clipes de referência
  ├── transcript_translated.json    ← tradução alinhada por segmento
  ├── subtitles_source.srt          ← legendas no idioma original
  ├── subtitles_target.srt          ← legendas no idioma de destino
  ├── dub_track.wav                 ← faixa de áudio dublada
  ├── manifest.json                 ← relatório completo da execução
  └── tts/ref_SPK_01.wav …          ← amostra da voz clonada de cada ator
```

---

## 6. Arquitetura do código

```
dublador/
  config.py      caminhos, idiomas, DubladorConfig, descoberta de ferramentas
  hardware.py    detecção de CPU/GPU/memória/disco + auto-ajuste e estimativa de tempo
  compat.py      shims de compatibilidade do host (torchaudio ↔ soundfile)
  utils.py       logging, execução de subprocessos, JSON, cache de etapas
  media.py       wrappers de ffmpeg/ffprobe, ajuste de tempo, mix e mux
  schema.py      modelos de dados (Transcript, Segment, SpeakerProfile, Word)
  transcribe.py  Whisper (faster-whisper) + timestamps por palavra
  speakers.py    diarização + seleção de clipes de referência
  translate.py   tradução por segmento com cache e fallback
  tts.py         clonagem de voz e síntese com F5-TTS
  assemble.py    remontagem da linha do tempo (numpy) + separação opcional
  lipsync.py     Wav2Lip (rede + S3FD) com degradação elegante
  pipeline.py    orquestrador das 7 etapas
  cli.py         interface de linha de comando + diagnóstico
  gui.py         servidor web local (API + gerenciador de trabalhos)
  static/
    index.html   interface (HTML/CSS/JS puro, sem CDN, funciona offline)
```

Ferramentas em `tools/`:

```
bootstrap.py        recria/repara todo o ambiente
download.js         downloader HTTPS (contorna o TLS quebrado do PowerShell)
pywheel.js          baixa um wheel do PyPI via API JSON
manual_install.py   instala pacotes que só existem como sdist (sem build backend)
make_test_video.py  gera vídeo de teste com diálogo multi-locutor (usa o F5-TTS)
test_f5_smoke.py    valida a síntese isoladamente e mede o custo real
test_gui.py         testa a interface de ponta a ponta (18 verificações)
test_pipes.py       descobre quais modos de subprocesso o sandbox permite
verify_output.py    confere se o vídeo final tem fala audível e no idioma certo
diag.py             mostra a similaridade entre vozes e a qualidade da diarização
probe.js            relata CPU, RAM e disco
```

---

## 7. Notas de ambiente (por que este projeto tem código "estranho")

Esta máquina tem três restrições que moldaram o código. Estão documentadas
porque qualquer uma delas pode reaparecer ao portar o projeto:

1. **O sandbox nega pipes de subprocesso.** Qualquer binário cujo `stdout` seja
   canalizado falha com `WinError 5 / Acesso negado`. Por isso
   `dublador/utils.py::run_command` redireciona a saída para um **arquivo
   temporário** e o lê de volta — **nunca use `subprocess.PIPE` neste projeto**.

2. **O TLS do .NET/PowerShell está quebrado** (o handshake morre em qualquer
   host HTTPS), mas o **OpenSSL do Python e o Node funcionam normalmente**. Por
   isso o bootstrap usa Node (`tools/download.js`) para baixar arquivos, e o pip
   funciona sem problemas.

3. **pip não consegue compilar sdists** (o build em isolamento precisa de
   pipes). Wheels funcionam; pacotes que só existem como sdist não. Dois deles
   são obrigatórios: `antlr4-python3-runtime==4.9.3` (exigido pelo
   `omegaconf`/`hydra-core`) e `encodec==0.1.1` (exigido pelo `vocos`). Ambos
   são Python puro, então `tools/manual_install.py` baixa o sdist, extrai e
   copia o pacote para o `site-packages`, sintetizando um `.dist-info` — sem
   invocar nenhum build backend.

4. **`torchaudio` ≥ 2.9** removeu o I/O nativo e passou a exigir **TorchCodec**,
   que não está instalado. Como o F5-TTS chama `torchaudio.load`, o módulo
   `dublador/compat.py` instala implementações alternativas baseadas em
   `soundfile` mantendo exatamente a mesma assinatura.

---

## 8. Limitações conhecidas

- **Sem separação de fontes.** No modo `--background duck` a música e os efeitos
  são preservados, mas **as vozes originais continuam audíveis em volume
  reduzido**, porque não há remoção de vocal. Use `none` (padrão) para uma
  dublagem limpa, ou instale o Demucs e use `separate` (muito lento em CPU).
- **Idiomas não latinos.** O F5-TTS foi treinado majoritariamente em inglês e
  chinês. Para japonês, coreano, russo, árabe etc. a pronúncia tende a piorar.
- **Lip sync não é perfeito.** O Wav2Lip re-anima apenas a região da boca e
  funciona melhor em *talking heads*; em cenas com muita oclusão ou vários
  rostos o resultado degrada.
- **A tradução por segmento perde contexto** entre falas longas. Nomes próprios
  podem variar; um glossário por job (`glossary`) está previsto na API.
- **Diante de muitas vozes parecidas** a diarização pode agrupar locutores.
  Ajuste `--max-speakers` e `--speaker-threshold`.
- **O sotaque viaja com a voz.** O F5-TTS clona o timbre a partir de um trecho
  do áudio original, então a fala no idioma de destino **mantém o sotaque do
  idioma de origem**. No teste com referência em inglês, o resultado em
  português soou como um falante de português com sotaque inglês — e o próprio
  Whisper, ao reouvir, chutou "inglês" com 60% de confiança, embora as palavras
  fossem claramente "Boa noite, o sistema está online". Isso é inerente à
  clonagem cross-lingual, não um defeito do pipeline. Para eliminar o sotaque
  seria preciso treinar/fine-tunar uma voz nativa por idioma.
- **A qualidade da transcrição limita tudo o que vem depois** — tradução,
  diarização e a própria dublagem. Em áudio com música alta ou ruído, suba para
  `--whisper-model medium` ou `large-v3`.

---

## 9. Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| `no speech was detected` | VAD agressivo ou áudio muito baixo. Tente `--no-vad` e `--whisper-model medium`. |
| `F5-TTS is not usable` | Rode `.\run.ps1 check`. Veja a seção 7 (sdists / torchaudio). |
| Tradução falhando | Sites de tradução bloqueados/limitados. Tente `--translator mymemory`. |
| Dublagem dessincronizada | Reduza `--nfe-step`? Não: ajuste `--max-stretch` ou desligue `--no-fix-duration`. |
| `WinError 5 / Acesso negado` | Você canalizou a saída de um binário. Use redirecionamento para arquivo. |
| Falta de espaço em disco | Só há ~12 GB livres. Apague `work\` de jobs antigos. |
| Muito lento | Esperado em CPU. Veja a tabela da seção 4. |
| `ABRIR-INTERFACE.cmd` fecha na hora | Veja a mensagem na janela; normalmente é o Python portátil ausente. Rode `tools\bootstrap.py`. |
| Interface não abre no navegador | Abra manualmente `http://127.0.0.1:8760/` (a porta aparece na janela preta). |
| “Não foi possível abrir a porta” | Outra instância já está rodando. Use `.\run.ps1 gui --port 8900`. |
| Botão “Dublar” dá erro 403 | Recarregue a página — o token muda a cada reinício do servidor. |
| Upload falha em arquivo grande | Limite de 4 GB, e confira o espaço em disco (só ~11 GB livres). |
| Trabalho aparece como “interrompido” | A interface foi fechada durante a execução. Rode de novo: continua de onde parou. |
