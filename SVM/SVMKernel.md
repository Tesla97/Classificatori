# SVM Kernel
Le macchine a vettori di supporto sono ovviamente classificatori lineari binari. Nascono con l'obiettivo di massimizzare il margine di separazione tra le classi. In particolare come è ben noto , l'individuazione di un margine decisionale lineare , passa attravero un problema di ottimo vincolato , più precisamente , il vettore dei pesi w , altro non è che soluzione del seguente problema di ottimo 

<p>
  min |w|</br>
  s.t</br>
  y(i) (w x(i) + w0 ) >= 1 , i = 1 ... N
</p>

(|w| intendiamo una qualsiasi norma) quindi di fatto minimizzo la norma del vettore w sotto il vincolo che gli esempi di classificazione vengano classificati in maniera corretta. In particolare gli esempi tali per le quali le precedenti diseguaglianze sono verificate all'eguaglianza prendono il nome di vettori di supporto.
Ora che cosa notiamo? Notiamo che se le classi non sono separabili linearmente , la regione ammissibile del precedete problema coincide con l'insieme vuoto. Il problema non ha soluzione.
Nel 1995 , Vapnik , con lo scopo di applicare le macchine SVM , a problemi di classificazione con classi non linearmente separabili , introdusse , con lo scopo di rilassare i vincolo , le variabili di slack , una per ogni esempio di addestramento

<p>
  min |w|</br>
  s.t</br>
  y(i) (w x(i) + w0 ) >= 1 - c(i), i = 1 ... N </br>
  c(i) >= 0
</p>

Ma in realtà , il motivo per cui le SVM sono così popolari nella computer science è che queste sono facilmente kernelizzabili per risolvere problemi di classificazione non lineare. Spieghiamoci meglio , e facciamolo con un esempio

Iniziamo la nostra discussione creando un semplice dataset di addestramento (ad hoc per l'esempio)

```bash
def createDatasetForSVMKernel(nEsempi,r1,r2,random_state,low,high,classe,intersection):
    np.random.seed(random_state)
    counter = 0
    list    = []
    y       = []
    if(intersection == 0):
        while(counter != nEsempi):
          X    = np.random.randint(low,high,size=nEsempi)
          Y    = np.random.randint(low,high,size=nEsempi)
          for i in range(0,nEsempi):
            if(np.power(X[i],2) + np.power(Y[i],2) > np.power(r1,2)):
                list.append([X[i],Y[i]])
                y.append(classe)
                counter += 1
        return list  , y
    while(counter != nEsempi):
          X    = np.random.randint(low,high,size=nEsempi)
          Y    = np.random.randint(low,high,size=nEsempi)
          for i in range(0,nEsempi):
            if(np.power(X[i],2) + np.power(Y[i],2) > np.power(r1,2) and np.power(X[i],2) + np.power(Y[i],2) < np.power(r2,2) ):
                list.append([X[i],Y[i]])
                y.append(classe)
                counter += 1
    return list  , y
```
questo metodo va a creare esempi di addestramento al di fuori di alcune circonferenze. In particolare andiamo a creare 20 esempi di addestramento come segue

```bash
lista_1 , y_1  = createDatasetForSVMKernel(nEsempi=10,r1=10,r2=0,random_state=1,low=-100,high=100,classe=1,intersection=0)
lista_2 , y_2  = createDatasetForSVMKernel(nEsempi=10,r1=5, r2=7,random_state=1,low=-50,high=50,classe=-1,intersection=1)
```
e andiamo a graficarli

<p align='center'>
  <img src='/img/nonLineare.png' width='50%'>
</p>

come notiamo non vi è nessuna esistenza di un iperpiano che riesca esattamente a dividere i campioni delle rispettive classi. Le classi non sono separabili linearmente. Una classica SVM non avrebbe nessuna soluzione (se non con l'introduzione delle variabili di slack).

L'idea alla base delle SVM Kernel è quello di proiettare i campioni in uno spazio a più elevata dimensionalità , andando a considerare generiche combinazioni delle caratteristiche di partenza. Ovvero , la costruzione di una mappa phi(x) , che aumenti la dimensionalità del campione , e sperando che attraverso questa proiezione i campioni delle rispettive classi siano linearmente separabili.
Nel costro caso scegliamo come mappa , la seguente

phi(x1,x2) = (x1,x2,x1^2 + x2^2)

e vediamo se con il passare da R^2 a R^3 per mezzo di questa trasformazione fa si che il precedente insieme di campioni risulti separabile da un iperpiano. Di seguito il codice per aumentare la dimensionalità del generico campione

```bash
# aumentare la dimensionalità del campione
    z_1 = np.power(class_1_x1,2) + np.power(class_1_x2,2)
    z_2 = np.power(class_2_x1,2) + np.power(class_2_x2,2)
    #plt.figure(2)
    #ax  = plt.axes(projection='3d')
    #ax.scatter(class_1_x1,class_1_x2,z_1,marker='o',color='red',label='1')
    #ax.scatter(class_2_x1,class_2_x2,z_2,marker='x',color='blue',label='-1')
    plt.grid()
    plt.legend(loc='best')
    # combinazione dati
    X = []
    y = []
    for i in range(0,10):
        X.append([class_1_x1[i],class_1_x2[i]])
        y.append(y_1[i])
    for i in range(0,10):
        X.append([class_2_x1[i],class_2_x2[i]])
        y.append(y_2[i])
    X = np.array(X)
    y = np.array(y)
```
e plottiamo il nuovo dataset di addestramento in questo spazio esteso (R^3). 

<p align='center'>
  <img src='/img/aumentoDimensione.png' width='60%'>
</p>

come notiamo adesso esiste un iperpiano che separa perfettamente le due classi. Addestriamo in questo nuovo spazio un classificatore e , all'arrivo dei dati di test , proiettiamo i dati di test su questo a dimensione più elevata e ritorniamo la classe di appartenenza relativa restituitaci dal classificatore addestrato in questo spazio esteso. Ovviamente il confine decisionale lineare nello spazio di partenza sarà ovviamente non lineare.

Il classificatore verrà addestrato sulla base dei campioni phi(x(i)) i = 1 ... N. Fare la mappatura (specialmente se i dati sono a elevata dimensionalità) non solo richiede abbastanza tempo , ma aumenta considerevolmente il tempo per effettuare operazioni (ad esempio prodotti scalari) tra i nuovi esempi di addestramento. Ad esempio (prendiamolo per buono) se la definizione dell'iperpiano avvenisse tramite ad esempio una Adaline , la determinazione dei pesi ottimi passa attraverso il calcolo dei seguenti prodotti scalari (equazioni normali)

<p align='center'>
phi(x(j))' phi(x(i))
</p>

con l'obiettivo di elimiare i costi computazionali aggiuntivi relativi alla mappatura nonchè al calcolo dei prodotti scalari precedenti , vengono introdotte le funzioni kernel. In particolare la funzione kernel più usata è la RBF (Radial Basis Function)

<p align='center'>
k(x(i), x(j)) = exp{ (-1/2sigma) (|x(i) - x(j)|) }
</p>

che cosa notiamo. Il prodotto scalare tra due è vettori e massimo quando i due vettori sono paralleli. Allo stesso modo la funzione RBF è massima quando x(i) è pari a x(j) , quindi di fatto costituisce una sorta di misura della similarità tra coppie di esempi di addestramento.

Il parametro gamma = 1/2sigma , è un parametro che andrà ottimizzato. Si riporta di seguito l'addestramento di una SVM Kernel (libreria sklearn) e il plotting della regione decisionale non lineare relativa.

```bash
# addestramento svm kernel
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
    svm = SVC(kernel='rbf',random_state=1,gamma=0.05,C=10.0)
    svm.fit(X_train,y_train)
```

regione decisionale non lineare relativa

<p align='center'>
  <img src='/img/confNonLinear.png' width='50%'>
</p>


