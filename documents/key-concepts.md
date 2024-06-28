# Concepts to cover during the presentation

1. Introduzione :
   1. Problemi al cuore causa principale di morte
   2. Obbiettivo di fornire un metodo accessibile a tutti per identificare possibili anomalie nel ciclo cardiaco, in modo da agire preventivamente
   3. Obbiettivo di fornire un supporto al medico per identificare problemi cardiaci in maniera più accurata
   4. Stato dell'arte ??
2. Dataset:
   1. Fonte
   2. struttura (Gruppo A,B,C)
   3. bilanciamento classi etc...
   4. Le diverse classi. Breve descrizione delle 5 classi presenti
3. Le features, breve descrizione delle features estratte (MFCC, CQT, Chroma ...) e del perché abbiamo scelto proprio queste
4. Estrazione delle features:
   1. Scelta dell'intervallo e del sampling rate (Parlare anche dei problemi identificati)
   2. Scelta dell numero ottimale di features per ogni tipo
5. Selezione delle features:
   1. Creazione dei filtri
   2. Selezione delle thresholds
   3. Risultati
6. Crezione dei modelli:
   1. Normal vs Abnormal (Orientato per il pubblico a scopo preventivo, quindi identificare solo se c'è o no un problema)
      1. ROC
      2. Models Performance with different accepted FPRs - Normal VS Rest
      3. Confusion Matrix
   2. Tutte le classi (Orientato per il Medico, per identificare il problema e dove si trova):
      1. Risk score
      2. Comparison on test set
7. Analisi modello ottimo
   1. Perchè Ensamble 2 è il migliore
   2. Confusion Matrix
   3. explainability:
      1. Permutation importance
      2. LIME, per identificare le features più significative per la predizione e risalire all'area dello spettrogramma più significativa (dove si trova l'anomalia)
      3. Permutation importance per ogni classe???
8. Other test:
   1. CNN on spectral
   2. CNN on waveform
   3. Ensemble with 3 different models
9. Limitazioni:
10. Classi fortemente sbilanciate e difficili da bilanciare (incrementare il dataset), siamo dovuto scendere a parecchi compromessi (scelta intervallo)
11. Impossibilità di creare il validation set per trovare gli hyperparametri, metriche di test sono biased
12. Valutare con esperti medici le tecniche utilizzate e la loro coerenza con gli aspetti medici (Soptrattutto LIME)
13. Perimentare altre features

## Paper Structure

- Abstract (250 words or less)


- Introduction (brief literature review, present the problem domain, identify
  gap(s) that need to be addressed, and a research question- how will you address the gap)


- Methods (source of data, methods for managing the data, data mining methods you used, with a justification for them)
      - Source of data
      -
- Results (listing and evaluation of the results)
- Discussion (position your work in the context of what has been done already,
  and an honest assessment of the limitations of your approach)
- Conclusion (overall impression of what you have done, and any work you
  propose for the future).
