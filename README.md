# Regressione-Logistica
La regressione logistica è un modello probabilistico impiegato per la risoluzione di problemi di classificazione lineari e binari , la variabile target , dunque , è una variabile dicotomica , cioè assume solo i valori 0 e 1.

La regressione logistica , come modello probabilistico , si fonda sul concetto di probabilità di appartenenza di un esempio ad una determinata classe

P{(y=1|x)} = p(x) 

P({y=0|x)} = 1 - p(x)

le probabilità sono grandezze varianti sull'intervallo [0,1] , quindi di fatto , non è possibile utilizzare un modello lineare per mettere in relazione il vettore delle caratteristiche x , con le probabilità di successo , perchè l'uscita di un generico modello lineare del tipo

                                                        y = mx + q
                                                                    
è sui reali. Al fine di esprimere una relazione lineare tra le caratteristiche e le probabilità , è necessario che p(x) sia l'argomento di una funzione che mappi l'intervallo [0,1] sui reali. La funzione maggiormente utilizzata è la funzione logit , definita come 

                                                     log (p(x)/(1-p(x)) 
                                                   
definita sull'intervallo [0,1] e invertibile. Quindi volendo esprimere una relazione lineare tra il vettore delle caratteristiche e i log-odds 

                                                   log(p(x)/(1-p(x)) = bx
                                                   
(si è messo nel vettore dei coefficienti il bias unit) , ma essendo interessati a determinare le probabilità di appartenenza , la funzione inversa è ben definita è vale

                                           p(x) = exp(bx)/(1+exp(bx)) = 1/(1+exp(-bx))
                                           
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
   <img src='img/sigmoide.png'>
</p>
                                                       
                                 


