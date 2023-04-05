# Regressione-Logistica
La regressione logistica è un modello probabilistico impiegato per la risoluzione di problemi di classificazione lineari e binari , la variabile target , dunque , è una variabile dicotomica , cioè assume solo i valori 0 e 1. Nella pratica non si è solamente interessati a predire l'etichetta delle classi , ma si vuole in qualche modo determinare la probabilità di appartenenza. Quindi non solo vogliamo stabilire a che classe appartiene un determinato esempio, ma siamo anche interessati a stimarne una probabilità di appartenenza.

La regressione logistica , come modello probabilistico , si fonda sul concetto di probabilità di appartenenza di un esempio ad una determinata classe
<p align='center'>
P{(y=1|x)} = p(x)
</br></br>
P({y=0|x)} = 1 - p(x)
</p>

le probabilità sono grandezze varianti sull'intervallo [0,1] , quindi di fatto , non è possibile utilizzare un modello lineare per mettere in relazione il vettore delle caratteristiche x , con le probabilità di successo , perchè l'uscita di un generico modello lineare del tipo
<p align='center'>
  y = mx + q
<p>
                                                                    
è sui reali. Al fine di esprimere una relazione lineare tra le caratteristiche e le probabilità , è necessario che p(x) sia l'argomento di una funzione che mappi l'intervallo [0,1] sui reali. La funzione maggiormente utilizzata è la funzione logit , definita come 
<p align='center'>
                                                     log (p(x)/(1-p(x)) 
</p>                                            
definita sull'intervallo [0,1] e invertibile. Quindi volendo esprimere una relazione lineare tra il vettore delle caratteristiche e i log-odds

<p align = 'center'>
    </br>log (p(x)/(1-p(x))  = bx
</p>

(si è messo nel vettore dei coefficienti il bias unit) , ma essendo interessati a determinare le probabilità di appartenenza , la funzione inversa è ben definita è vale

<p align = 'center'>
                                           p(x) = exp(bx)/(1+exp(bx)) = 1/(1+exp(-bx))
</p>          

nota in letteratura come funzione sigmoide , per via della sua forma ad S.

Scriviamo un semplice codice python per visualizzarla a schermo

```bash
import matplotlib.pyplot as plt
import numpy as np

# funzione sigmoide
def sigmoideFunction(x):
    return 1.0 / (1.0 + np.exp(-x))

# main
if __name__ == '__main__':
    x = np.arange(-10,10,0.1)
    y = sigmoideFunction(x)
    # plotting
    plt.figure(1)
    plt.plot(x,y)
    plt.grid()
    plt.show()
```
<p align='center'>
   <img src='/img/sigmoide.png' width = '50%'>
</p>

Come notiamo questa funzione prende in input delle combinazioni lineari delle caratteristiche (input della rete) e li mappa in valori compresi sull'intervallo [0,1] , interpretabili come la probabilità che il generico esempio con caratteristiche x appartenga alla classe y = 1.In phearticolare si noti che la funzione vale 0.5 , quando l'input della rete è zero. 

A questo punto possiamo impiegare una semplice funzione gradino (Heaviside) per convertire il valore di probabilità in un valore binario

       if (sigmoide(x) >= 0.5) 
           y = 1
       else 
           y = 0

Per determinare il vettore dei besi b , si utilizza un semplice stimatore a massima verosimiglianza. Ipotizzando l'indipendenza tra i campioni di addestramento , il funzionale (verosimiglianza) che si intende massimizzare

<p align='center'>
  <img src='/img/funzionale.png' width='40%'> 
</p>

ovviamente basta moltiplicarlo per -1 per passare ad un problema di minimizzazione. Ma facciamo un attimo una piccola analisi sullo i-esimo termine della sommatoria

<p align='center'>
  <img src='/img/singolotermine.png' width='30%'> 
</p>

notiamo che vale log(sigmodide(x)) quando y_i = 1 altrimenti log(1-sigmoide(x)).Di seguito è riportato il codice per il tracciamento del funzionale di costo nei due casi

```bash
# funzionale di costo y_i = 1
def costoPrimo(x):
    return np.log(sigmoideFunction(x))

# funzionale di costo y_i = 0
def costoSecondo(x):
    return np.log(1 - sigmoideFunction(x))

# main
if __name__ == '__main__':
    x  = np.arange(-10,10,0.1)
    c1 = costoPrimo(x)
    c2 = costoSecondo(x)
    t  = sigmoideFunction(x)
    plt.figure(1)
    plt.plot(t,c1,label='y=1')
    plt.plot(t,c2,linestyle='--',label='y=0')
    plt.show()
```
passiamo ora a graficare il risultato per camirne meglio il comportamento dei singoli contributi.

<p align='center'>
  <img src='/img/funzionaliConfronto.png' width='50%'>
</p>

notiamo che nei rispettivi casi il costo è massimo se prediciamo correttamente l'appartenenza dell'esempio alla classe corrispondente. Al contrario se la predizione è errata il costo va a meno infinito.Le predizioni errate vengono dunque pesate con un costo sempre minore.


<h1> Implementazione con Scikit-Learn </h1>
Vediamo ora una implementazione del precedente algoritmo di classificazione con riferimento ad un classico dataset , il dataset Iris. In particolare faremo uso della libreria scikit-learn , la quale fornisce una implementazione ben ottimizzata , che di default supporta anche situazioni multiclasse (tecnica OvR or OvA).

Iniziamo innanzitutto con caricare il dataset Iris , contenuto già nella libraria scikit-learn.

```bash
from sklearn import datasets

if __name__ == '__main__':
    # caricamento dataset
    ds = datasets.load_iris()
    # caricamento esempi di addestramento
    X  = ds.data[:,[0,1]]
    y  = ds.target
```

in particolare notiamo come carichiamo solo le righe corrispondendi a 2 delle 4 totali caratteristiche , questo per scopi illustrativi.In particolare stiamo facendo riferimento alla lunghezza ed alla larghezza del sepalo. Quello che facciamo ora e implementare la cross validation , e quindi andiamo a suddividire i dati di addestramento in due parti , una per l'addestramento e l'altra per la valutazione del modello ottenuto.

```bash
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # caricamento dataset
    ds = datasets.load_iris()
    # caricamento esempi di addestramento
    X  = ds.data[:,[0,1]]
    y  = ds.target
    # cross validation
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
```
Quindi lo 80% dei dati verranno utilizzati per l'addestramento ed il restante 20% per la validazione del modello. Ai fini di massimizzare le prestazioni dell'algoritmo , andiamo a standardizzare i dati

```bash
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # caricamento dataset
    ds = datasets.load_iris()
    # caricamento esempi di addestramento
    X  = ds.data[:,[0,1]]
    y  = ds.target
    # cross validation
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
    # standardizzazione
    sc = StandardScaler()
    sc = sc.fit(X_train)
    X_train_std= sc.transform(X_train)
    X_test_std = sc.transform(X_test) 
```
In particolare quello che abbiamo fatto è standardizzare i dati , e quindi visibili come realizzazioni di una normale standard a media nulla e varianza unitaria. In particolare si noti come la standardizzazione sui dati di test viene fatta in accordo a media e varianza calcolati sui dati di addestramento. Questo ovviamente , per permetterne un cofronto.

Iniziamo con plottare i dati di addestramento al fine di verificare l'esistenza di un iperpiano (retta) di separazione

```bash
    plt.figure(1)
    plt.scatter(X[:50,0],X[:50,1],marker='o',color='red',label='Setosa')
    plt.scatter(X[50:100,0],X[50:100,1],marker='x',color='blue',label='Versicolor')
    plt.scatter(X[100:150,0],X[100:150,1],marker='^',color='green',label='Virginica')
    plt.legend(loc='upper left')
    plt.show()
```
<p align='center'>
  <img src='/img/verificaEsistenzaIperpiano.png' width='50%'>
</p>

Come possiamo notare i dati non sono separabili linearmente. Procediamo dunque con l'addestramento di un modello a regressione logistica , come di seguito riportato

```bash
# regressione logistica
lr = LogisticRegression(C = 100.0 , random_state= 1 , solver='lbfgs' , multi_class='ovr')
lr.fit(X_train_std,y_train)
```

come ultimo passo andiamo a valutare le prestazioni sull'insieme dei dati di test

```bash
# predizione
y_pred = lr.predict(X_test_std)
errors = (y_pred != y_test).sum()
print('Errori Classificazione: %d' %errors)
```
ovviamente come visto , vi saranno 11 errori in classificazione , il problema è dovuto al fatto che le caratteristiche selezionate non sono adatte. Si riprovi l'addestramento facendo riferimento alle altre 2 caratteristiche.

Tracciamo infine le regioni decisionali del modello

<p align='center'>
  <img src='/img/regioniDecisionali.png' width='50%'>
</p>

Ricordiamo che la regressione logistica come modello probabilistico , si basa sul concetto di probabilità di appartenenza ad una determinata classe. Andiamo a graficare le probabilità di appartenenza dei 30 esempi di addestramento alle rispettive 3 classi

```bash
print(lr.predict_proba(X_test_std))
```

il risultato dovrebbe essere il seguente

```bash
[[2.47557367e-13 3.00504232e-01 6.99495768e-01]
 [7.92208379e-01 1.98744537e-01 9.04708383e-03]
 [3.41215074e-02 9.23311163e-01 4.25673295e-02]
 [8.93604503e-01 1.00374284e-01 6.02121330e-03]
 [9.12248256e-01 7.71628826e-02 1.05888615e-02]
 [9.08527849e-01 7.71383832e-02 1.43337679e-02]
 [1.06931770e-03 2.54346840e-01 7.44583843e-01]
 [5.25289570e-09 4.75324741e-01 5.24675253e-01]
 [1.94450786e-03 8.06907880e-01 1.91147613e-01]
 [6.83937839e-11 3.66523263e-01 6.33476737e-01]
 [9.48132993e-01 2.38962286e-02 2.79707785e-02]
 [1.05455013e-05 8.73919566e-01 1.26069889e-01]
 [9.09504342e-10 2.88193033e-01 7.11806966e-01]
 [2.51848246e-09 6.80569978e-01 3.19430019e-01]
 [8.47480247e-06 7.99036250e-01 2.00955275e-01]
 [9.46931514e-01 4.91691318e-03 4.81515732e-02]
 [1.40105352e-11 2.66677470e-01 7.33322530e-01]
 [1.47386718e-06 5.61253610e-01 4.38744917e-01]
 [1.20094928e-01 8.58600467e-01 2.13046050e-02]
 [2.17904202e-05 4.88548300e-01 5.11429909e-01]
 [7.25739122e-08 4.07517066e-01 5.92482862e-01]
 [3.01114270e-02 7.42279530e-01 2.27609043e-01]
 [9.34199325e-01 5.83892883e-02 7.41138630e-03]
 [7.20897588e-01 2.77075105e-01 2.02730700e-03]
 [6.42669894e-08 3.39000335e-01 6.60999601e-01]
 [2.87136203e-01 6.95349840e-01 1.75139565e-02]
 [9.48132993e-01 2.38962286e-02 2.79707785e-02]
 [7.55125523e-01 2.38060628e-01 6.81384956e-03]
 [6.30548429e-05 2.63737313e-01 7.36199633e-01]
 [4.66856383e-09 4.17252862e-01 5.82747134e-01]]
```
