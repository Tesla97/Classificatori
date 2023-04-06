# Macchine a Vettori di Supporto (SVM)
Ancora una volta come la regressione logistica , il percettrone e così via , le macchine a vettori di supporto rientrano nella categoria di algoritmi di apprendimento per problemi di classificazione lineari binari. Però a differenza di un tradizionale algoritmo di apprendimento , il cui obiettivo è quello di scegliere il vettore dei coefficienti dell'iperpiano in modo da minimizzare un funzionale di costo quadratico (RMS) , le SVM si pongono come obiettivo la massimizzazione del margine di separazione tra le classi. Questo ragionamento premia molto per quando riguarda la generalizzazione del classificatore.

### Introduzione alle SVM
Supponiamo di considerare un generico problema di classificazione , nella ipotesi che le classi , siano separabili da un confine decisionale lineare. Senza perdita di generalità al discorso supponiamo che le classi rientrano nell'insieme {1,-1}. Supponiamo l'esistenza della coppia (m,q) tale che 

<p align='center'>
  m x(i) + q >=  1      if y(i) =  1</br>
  m x(i) + q <= -1      if y(i) = -1 
</p>

in particolare si noti che le precedenti condizioni possono essere riespresse in modo più compatto come segue

<p align='center'>
  y(i) ( m x(i) + q ) - 1 >= 0
</p>

dove la coppia (x(i),y(i)) fa riferimento allo i-esimo campione di addestramento fornito per l'apprendimento. Supponiamo l'esistenza di esempi di addestramento (x(i),y(i) tale che le precedenti diseguaglianze siano verificate all'uguaglianza , questi sono defini i vettori di supporto. Si ricordi che l'obiettivo è la massimizzazione del margine , ovvero la massimizzazione della distanza di ciascun vettore di supporto all'iperpiano di separazione.

La distanza punto retta è definita come segue

<p align='center'>
  dist(x,(m,q)) = ||mx + q|| / ||m|| 
</p>

in particolare con riferimento ai vettori di supporto indipendemente quale sia la classe di appartenenza , otteniamo

<p align='center'>
  dist(x,(m,q)) = 1 / ||m||
</p>

la massimizzazione della precedente funzione obiettivo , è del tutto equivalente alla risoluzione del seguente problema di ottimizzazione sotto il vincolo che gli esempi vengano classificati correttamente , cioè soluzione del seguente problema

<p align='center'>
  min ||w|| </br>
  y(i)(mx(i)+q) >= 1  , i = 1 ...  N
</p>
