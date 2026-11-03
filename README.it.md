<p align="center">
  <img src="custom_components/snapshot_classifier/brand/icon.png" width="96" alt="Icona Snapshot Classifier">
</p>

# Snapshot Classifier per Home Assistant

🇬🇧 [Read in English](README.md)

Classifica lo snapshot di qualsiasi telecamera di Home Assistant con un modello di
immagini addestrato da te, nel browser, con
[Google Teachable Machine](https://teachablemachine.withgoogle.com/). Niente
cloud, niente add-on e niente TensorFlow: il modello gira in locale dentro Home
Assistant.

Uso tipico: *"Nel mio posto auto c'è **la mia auto**, **un'altra auto** o è
**vuoto**?"*. Ma va bene qualsiasi domanda a cui si risponde guardando una foto:
"il garage è aperto?", "i bidoni sono fuori?", "il cancello è chiuso?".

## Funzionalità

- Configurazione tutta da interfaccia: scegli la telecamera, carichi lo `.zip`
  esportato da Teachable Machine, indichi quale classe significa "vuoto".
- Entità per ogni classificatore: **Stato** (classe migliore o *Incerto*),
  **Sicurezza**, **Occupato**, **&lt;classe&gt; presente**, pulsante **Analizza ora** e
  l'evento **Risultato incerto**.
- Analizza un nuovo snapshot quando cambia un attributo della telecamera (es.
  `timestamp` delle telecamere ring-mqtt), a intervallo fisso o solo a richiesta.
- Ritaglio facoltativo, così il modello guarda solo la parte di immagine che
  interessa, più un servizio per ritagliare allo stesso modo le foto di addestramento.
- Salvataggio facoltativo degli snapshot incerti in una cartella, per sapere quali
  foto aggiungere all'addestramento.
- Sostituzione del modello con uno riaddestrato dalle opzioni, senza perdere le entità.
- Quanti classificatori vuoi (più telecamere, o più posti sulla stessa telecamera
  con ritagli diversi).
- Funziona su ogni installazione di Home Assistant, compreso Home Assistant OS su
  Raspberry Pi / Home Assistant Yellow / Green (aarch64): l'inferenza è in puro
  `numpy`, l'unico requisito è il parser Python puro
  [`tflite`](https://pypi.org/project/tflite/).

## Installazione

### HACS (consigliato)

1. HACS → ⋮ → **Repository personalizzati** → aggiungi
   `https://github.com/mauro-midolo/snapshot-classifier-home-assistant`, tipo
   **Integrazione**.
2. Cerca **Snapshot Classifier** in HACS e scaricalo.
3. Riavvia Home Assistant.

### Manuale

Copia `custom_components/snapshot_classifier` nella cartella `custom_components`
della configurazione di Home Assistant e riavvia.

## 1. Raccogli le foto di addestramento

Il modello è buono quanto le sue foto. Devono venire **dalla stessa telecamera,
con la stessa inquadratura**, in tutte le situazioni da riconoscere.

Salva uno snapshot periodicamente per qualche giorno (giorno, notte, pioggia,
sole, fari…). Per esempio ogni 15 minuti in `/media/parcheggio`:

```yaml
alias: Parcheggio - salva snapshot per addestramento
triggers:
  - trigger: time_pattern
    minutes: "/15"
actions:
  - action: camera.snapshot
    target:
      entity_id: camera.mia_telecamera
    data:
      filename: "/media/parcheggio/{{ now().strftime('%Y%m%d_%H%M%S') }}.jpg"
mode: single
```

> Le telecamere a batteria (es. Ring via ring-mqtt) aggiornano lo snapshot solo
> ogni tanto; se la telecamera lo permette, chiedine prima uno nuovo.

Scarica le foto sul computer (Browser multimediale, condivisione Samba, SSH…) e
dividile in una cartella per classe, es. `mia_auto/`, `altre/`, `vuoto/`.

### Facoltativo: ritaglio

Se la telecamera inquadra molto più del necessario (es. il posto auto è solo la
parte in basso a sinistra), usa un ritaglio: migliora molto la precisione.

- Il ritaglio è un riquadro in frazioni dell'immagine: sinistra (`x`), alto (`y`),
  larghezza (`width`), altezza (`height`), tutti tra 0 e 1. Esempio: il quarto in
  basso a sinistra è sinistra 0, alto 0.5, larghezza 0.5, altezza 0.5. Misuralo su
  uno snapshot con un visualizzatore di immagini: `sinistra = pixel_sinistro / larghezza_immagine`, ecc.
- Il ritaglio è applicato **prima** del quadrato centrale e del ridimensionamento a
  224×224 che fa Teachable Machine. Il riquadro può essere un rettangolo qualsiasi,
  ma ciò che interessa deve stare nel suo quadrato centrale.
- **Le foto di addestramento devono essere ritagliate esattamente allo stesso
  modo.** Usa il servizio `snapshot_classifier.export_crops`:

```yaml
action: snapshot_classifier.export_crops
data:
  source: /media/parcheggio
  destination: /media/parcheggio_ritagli
  left: 0
  top: 0.5
  width: 0.5
  height: 0.5
```

Nel servizio i campi si chiamano `left`/`top` (non `x`/`y`: alcuni editor YAML
leggono `y:` come `true`). Quando il classificatore esiste puoi usare
`config_entry_id` al posto dei numeri per riusare il suo ritaglio. Il servizio restituisce quante foto ha esportato,
saltato (già presenti, a meno di `overwrite: true`) e non riuscito a leggere.

## 2. Addestra il modello su Teachable Machine

1. Apri <https://teachablemachine.withgoogle.com/> e clicca **Get Started** (Inizia).
2. Scegli **Image Project** → **Standard image model** (non "Embedded image model").
3. Crea **una classe per ogni situazione** e rinominala cliccando sul nome
   ("Class 1" → `Jeep`, "Class 2" → `Altre`, aggiungi una classe `Vuoto`). I nomi
   delle classi diventano gli stati del sensore *Stato*, quindi scegli nomi che ti
   piace vedere in Home Assistant.
4. Per ogni classe clicca **Upload** → **Choose images from your files** (o
   trascinale) e aggiungi le foto di quella classe (quelle ritagliate, se usi un
   ritaglio). Non usare il pulsante Webcam: le foto devono venire dalla tua telecamera.
5. Consigli per un buon modello:
   - Tieni le classi **bilanciate**: un numero simile di foto per classe (almeno
     30–50 ciascuna; di più è meglio). Una classe con pochissime foto verrà
     riconosciuta raramente (es. 139/58/6 è molto sbilanciato).
   - Copri tutte le condizioni: giorno/notte (infrarossi), pioggia, ombre, neve,
     persone che passano, auto parcheggiata un po' diversa, portiere aperte…
   - Non mettere la stessa foto in due classi e lascia fuori le foto ambigue.
6. Clicca **Train Model** e attendi (tieni la scheda del browser in primo piano).
   I valori predefiniti (50 epoche, batch 16, learning rate 0,001) vanno bene; in
   **Advanced** puoi aumentare le epoche se i risultati sono scarsi.
7. Provalo nel pannello **Preview**: imposta Input su **File** e trascina alcune
   foto *non* usate per l'addestramento.
8. Clicca **Export Model** → scheda **Tensorflow Lite** → Model conversion type
   **Floating point** (virgola mobile) → **Download my model**. La conversione
   richiede fino a un minuto; ottieni `converted_tflite.zip` con
   `model_unquant.tflite` e `labels.txt`.
   > ⚠️ I modelli "Quantized" ed "EdgeTPU" **non sono supportati**; la configurazione
   > ti dirà di esportare in virgola mobile.
9. Facoltativo: **salva il progetto** (☰ → *Save project to Drive* o *Download
   project as file*) per aggiungere foto e riaddestrare in futuro.

## 3. Aggiungi l'integrazione

**Impostazioni → Dispositivi e servizi → Aggiungi integrazione → Snapshot Classifier**:

1. **Nome** (es. `Parcheggio`), **Telecamera**, **Modello Teachable Machine**
   (carica `converted_tflite.zip`).
2. **Classi** (entrambe facoltative):
   - *Classe che significa "vuoto"* → crea il sensore **Occupato**.
   - *Classe da seguire* (es. `Jeep`) → crea il sensore **Jeep presente**.

Se la telecamera ha l'attributo `timestamp`, il classificatore analizza ogni nuovo
snapshot, altrimenti controlla ogni 60 secondi. Si cambia nelle opzioni.

## Opzioni

| Opzione | Predefinito | Descrizione |
| --- | --- | --- |
| Telecamera | scelta all'inizio | La telecamera da analizzare. Usa quella che ha scattato le foto di addestramento (es. la telecamera *Snapshot* di ring-mqtt, non *Live view*). |
| Soglia di sicurezza | 70 % | Sotto questa probabilità lo *Stato* è `Incerto`. |
| Quando analizzare | attributo se esiste `timestamp`, altrimenti intervallo | *Quando cambia l'attributo della telecamera*, *A intervallo fisso*, *Solo manualmente*. |
| Attributo da osservare | `timestamp` | Attributo il cui cambiamento significa "nuovo snapshot"; deve esistere sulla telecamera. Non usare `entity_picture`/`access_token`: cambia ogni 5 minuti anche senza nuova immagine. |
| Intervallo | 60 s | Intervallo di controllo. Le immagini identiche non vengono rianalizzate. |
| Classe "vuoto" / Classe da seguire | – | Vedi sopra. |
| Ritaglio | immagine intera | Sinistra (x), alto (y), larghezza, altezza tra 0 e 1. |
| Cartella per gli snapshot incerti | – | es. `/media/snapshot_classifier/incerti`. Deve essere una cartella consentita (`/media` lo è di default, le altre vanno in [`allowlist_external_dirs`](https://www.home-assistant.io/integrations/homeassistant/#allowlist_external_dirs)). |
| Sostituisci il modello | – | Carica uno `.zip` riaddestrato. Se le classi sono cambiate ti chiede di sceglierle di nuovo. |

## Entità

Per un classificatore chiamato *Parcheggio* (con Home Assistant in italiano gli
ID seguono i nomi tradotti, es. `sensor.parcheggio_stato`):

| Entità | Descrizione |
| --- | --- |
| Stato | Classe migliore, o `Incerto` sotto soglia. Attributi: `probabilities` (% per classe), `best_class`, `threshold`, `snapshot_time`, `analyzed_at`, `inference_seconds`. |
| Sicurezza | Probabilità della classe migliore (%). |
| Occupato | `on` quando la classe "vuoto" ha probabilità inferiore al 50 %. |
| &lt;classe&gt; presente | `on` quando la classe seguita ha probabilità del 50 % o più. |
| Analizza ora | Analizza lo snapshot attuale (anche se invariato). |
| Risultato incerto | Evento `uncertain` con `best_class`, `confidence`, `probabilities`, `snapshot_time` e `image_path` (se salvato). |

Occupato / presente usano le probabilità grezze, quindi funzionano anche quando lo
stato è incerto (es. 45 % Jeep, 40 % altre → occupato, ma Jeep non presente).

Esempio: notifica dei risultati incerti per migliorare le foto di addestramento.

```yaml
alias: Parcheggio - snapshot incerto
triggers:
  - trigger: state
    entity_id: event.parcheggio_risultato_incerto
    not_from: unavailable
actions:
  - action: notify.mobile_app_mio_telefono
    data:
      message: >
        Non sono sicuro: {{ trigger.to_state.attributes.best_class }}
        ({{ trigger.to_state.attributes.confidence }} %)
```

## Riaddestramento

Aggiungi le nuove foto (per esempio dalla cartella degli incerti) al progetto
Teachable Machine, addestra, esporta di nuovo in virgola mobile e carica lo zip da
**Impostazioni → Dispositivi e servizi → Snapshot Classifier → Configura →
Sostituisci il modello**.

## Come funziona

- Immagine dalla telecamera (`camera.async_get_image`) → ritaglio → quadrato
  centrale → ridimensionamento bilineare all'ingresso del modello (224×224) → pixel
  scalati in [-1, 1] → modello → probabilità. È lo stesso preprocessing di Teachable Machine.
- TensorFlow Lite / LiteRT e ONNX Runtime non hanno wheel per il Python Alpine
  (musl) di Home Assistant, quindi l'integrazione include un piccolo interprete
  TFLite scritto in `numpy`. Supporta le operazioni dei modelli di immagini di
  Teachable Machine (`PAD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MEAN`,
  `FULLY_CONNECTED`, `SOFTMAX`) e qualcuna in più (`MUL`, `RESHAPE`,
  `CONCATENATION`, pooling, attivazioni). I test lo confrontano con il runtime
  ufficiale LiteRT.
- Il modello viene letto una sola volta all'avvio; l'inferenza (≈30 ms su un PC,
  ben sotto il secondo su Raspberry Pi 4/CM4) gira nell'executor, mai nell'event loop.
- Il modello caricato è salvato in `config/.storage/snapshot_classifier/<id>/`
  (incluso nei backup, cancellato quando rimuovi l'integrazione).

## Risoluzione dei problemi

- **"Modello quantizzato"**: esporta di nuovo scegliendo *Floating point*.
- **Sempre incerto / classe sbagliata**: più foto e più bilanciate; verifica che le
  foto di addestramento abbiano lo stesso ritaglio del classificatore.
- **Non si aggiorna mai**: controlla l'opzione *Quando analizzare*. Con un
  attributo, guarda in Strumenti per sviluppatori → Stati quale attributo cambia a
  ogni nuovo snapshot.
- Log di debug:

  ```yaml
  logger:
    logs:
      custom_components.snapshot_classifier: debug
  ```

## Licenza

[MIT](LICENSE). Teachable Machine è un prodotto Google; questo progetto non è
affiliato né approvato da Google.
