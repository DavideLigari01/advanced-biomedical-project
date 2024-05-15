## To do

1. ** New feature**:
    - [ ] CQT
    - [ ] Spectrogram 

2. **Engineering**:
    - Automate the feature extraction process
    - One model per feature


4. **Improve performance**:
   - [ ] Murmirs Normal and Extrastoles are often missclassified, identify additional features to discriminate them

Flusso:

1. Feature selection:
   1. One model per feature, scelgo diversi modelli e senza bilanciamento, e li alleno per ogni feature per vedere quale feature è più discriminante
   2. Feature correlation: già fatto

2. Tipo di bilanciamento, scelgo un modello a caso ed lo alleno nelle seguenti condizioni
   1. No balancing
   2. Prior balancing
   3. Posterior balancing
   4. Both balancing

3. Model selection:
   alleno diversi modelli con le feature selezionate e con il bilanciamento scelto:
   1. Logistic regression
   2. Random forest
   3. SVM
   4. XGBoost
   5. LightGBM
   6. CatBoost
   7. MLP
