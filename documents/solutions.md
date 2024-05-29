# Problema risultati alti ma biased
## Possibili soluzioni
1. finta di niente -> da evitare
2. generare dati sintetici sul test -> Non funziona
3. generare dati sintetici SU tutti i dati e poi fare split 
4. diminuire intervallo a 1s, non migliore però buon compromesso
5. lavorare sui dati row e provare ensemble di modelli
6. trasformare il problema in uno binario (Artifact o no e normale e anormale)
7. CNN per classificare i dati