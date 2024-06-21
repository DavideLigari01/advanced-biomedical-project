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
8. Limitazioni:
   1. Classi fortemente sbilanciate e difficili da bilanciare (incrementare il dataset), siamo dovuto scendere a parecchi compromessi (scelta intervallo)
   2. Impossibilità di creare il validation set per trovare gli hyperparametri, metriche di test sono biased
   3. Valutare con esperti medici le tecniche utilizzate e la loro coerenza con gli aspetti medici (Soptrattutto LIME)
   4. Perimentare altre features
