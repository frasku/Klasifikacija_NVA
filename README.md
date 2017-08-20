# Klasifikacija imenica, glagola i pridjeva
> Klasifikacija je provedena u programskom jeziku Python. Korišten je modul nltk za klasifikaciju te dva klasifikatora: Naivni Bayes i Maksimalna entropija. Ukupna količina riječi je 2,221,736 od kojih je 75% korišteno za trening, a ostatak 555,434 za testiranje.

###  Usporedba rezultata kod različitih klasifikatora korištenjem različitih osobina.
##### Preciznost: % odabranih elemenata koji su točni? IP/IP+LP
##### Opoziv:% točnih elemenata koji su odabrani? IP/IP+LN
##### F1 mjera je kombinirana mjera koja procjenjuje Preciznost/Opoziv (težinska harmonijska sredina)

##### 4 moguća slučaja:
  - IP - istina pozitivna
  - LP - laž pozitivna
  - LN - laž negativna
  - IN - istina negativna


| |točno | nije točno  |
|---------|-----|----|
|odabrano      | IP | LP |
| nije odabrano      | LN      |   IN| 
 
> Stupac 'točno' označava količinu koju klasifikator izbaci po  trenutnoj vrsti riječi, može biti pozitivno ili negativno, npr. za skup imenica, klasifikator je pogodio da su to imenice (IP) te u suprotnom krivo je pogodio, da nisu imenice (LN). Drugi stupac označava sve ostale vrste riječi, npr. kada klasifikator za sve ostale vrste riječi izbaci da su imenice (LP), te kada klasifikator za sve ostale vrste riječi izbaci da nisu imenice (IN).

## 1. Osobina

```sh
def osobine(word):
    if len(word)>10:return{'First and last letter':word[-6:]}
    if len(word)==10:return{'First and last letter':word[-5:]}
    if len(word)==5:return{'First and last letter':word[-3:]}
    if len(word)==4:return{'First and last letter':word[0]+word[-1]}     
    if len(word)>5 and len(word)<10: return {'Last 3 letters':word[-4:]}
    if len(word)<=3: return {'Last 2 letters':word[-2:]}
```
#### Naivni Bayes (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:

|  | imenica | ostalo |                                
|-----|-------|-----|
|odabrana imenica      | 41 | 4 |
|odabran ostatak      | 5     |  8 |
 
 
 
| |  pridjev | ostalo|  
|-----|------|----|
|odabran pridjev      | 130 | 12 |
|odabran ostatak      | 4     |   1|
 
 
|     | glagol | ostalo |
|-----|-------------|-----|
|odabran glagol      | 12 | 1|
|odabran ostatak      | 8      |   8 |


###### Preciznost imenice: 0,91 % -------Opoziv imenice: 0,46 %---------F1 mjera: 0,61
###### Preciznost pridjevi: 0,92 % -------Opoziv pridjevi: 0,99 %---------F1 mjera: 0,95
###### Preciznost glagoli: 0,92 %-------Opoziv glagoli: 0,6 %---------F1 mjera: 0,73
#### Maksimalna entropija (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:

|      | imenica | ostalo   |                     
|-------------|-------------|-----|
|odabrana imenica      | 41 | 4 |
|odabran ostatak      | 5     |  8 |

|    | pridjev | ostalo  |
|---------|--------|-----|
|odabran pridjev      | 130 | 12 |
|odabran ostatak      | 4     |   1|
 
 
|     | glagol | ostalo  |
|-------------|-------------|-----|
|odabran glagol      | 12 | 1 |
|odabran ostatak      | 8      |   8 |
 
###### Preciznost imenice: 0,91 %-------Opoziv imenice: 0,46 %---------F1 mjera: 0,61
###### Preciznost pridjevi: 0,92 %-------Opoziv pridjevi: 0,99 %---------F1 mjera: 0.95
###### Preciznost glagoli: 0,92 %------- Opoziv glagoli: 0,6 %---------F1 mjera: 0,73
## 2. Osobina

```sh
def osobine(word):
    if len(word)==4:return{'First and last letter':word[0]+word[-1]}    
    if len(word)>4: return {'Last 3 letters':word[-3:]}
    if len(word)<=3: return {'Last 2 letters':word[-2:]}
```
#### Naivni Bayes (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:

|    | imenica | ostalo|                                 
|-------------|-------------|-----|
|odabrana imenica      | 32 |2|
|odabran ostatak      | 5  | 17| 
 
|      | pridjev | ostalo|  
|-------------|-------------|-----|
|odabran pridjev      | 131| 21 |
|odabran ostatak      |3    |  0|

|    | glagol | ostalo |
|-------------|-------------|-----|
|odabran glagol      | 13 | 1 | 
|odabran ostatak      |16    | 7 |
 
###### Preciznost imenice: 0,94 %-------Opoziv imenice: 0,65 %---------F1 mjera: 0,77
###### Preciznost pridjevi: 0,86 %-------Opoziv pridjevi: 100 %---------F1 mjera: 0,93
###### Preciznost glagoli: 0,93 %------- Opoziv glagoli: 0,65 %---------F1 mjera: 0,77

#### Maksimalna entropija (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:
|    | imenica | ostalo |                                
|------|---------|----|
|odabrana imenica      | 32 |2|
|odabran ostatak      | 5  | 17 |
 
|      | pridjev | ostalo  |
|----|-------------|-----|
|odabran pridjev      | 131| 21 |
|odabran ostatak      |3    |  0 |

|     | glagol | ostalo |
|-----|-------|-----|
|odabran glagol      | 13| 1 |
|odabran ostatak      |16    | 7 |
###### Preciznost imenice: 0,94 %------- Opoziv imenice: 0,65 %---------F1 mjera: 0,77
###### Preciznost pridjevi: 0,86 %------- Opoziv pridjevi: 100 %---------F1 mjera: 0,93
###### Preciznost glagoli: 0,93 %-------Opoziv glagoli: 0,65 %---------F1 mjera: 0,77

## 3. Osobina

```sh
def osobine(word):
    return {'Last 3 letters':word[-3:]}  

```
#### Naivni Bayes (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:
|     | imenica | ostalo |                     
|-------------|------------|-----|
|odabrana imenica      |31 | 2|
|odabran ostatak      | 5     |  18|

|     | pridjev | ostalo |
|---------|------|-----|
|odabran pridjev      | 131 | 21|
|odabran ostatak      |4     |0 |
 
|    | glagol | ostalo  |
|-------------|-------------|-----|
|odabran glagol      | 13 |2 |
|odabran ostatak      | 16     | 7 |
###### Preciznost imenice: 0,94 %-------Opoziv imenice:  0,63 %---------F1 mjera: 0,75
###### Preciznost pridjevi: 0,86 %------- Opoziv pridjevi: 100 %---------F1 mjera: 0,93
###### Preciznost glagoli: 0,87 %-------Opoziv glagoli:  0,65 %---------F1 mjera: 0,74


#### Maksimalna entropija (preciznost, opoziv i F1 mjera)
###### Tablica slučaja za imenice, pridjeve i glagole:
|      | imenica | ostalo   |                             
|-------------|------------|-----|
|odabrana imenica      |31 | 2|
|odabran ostatak      | 5     |  18|

|   | pridjev | ostalo |
|----------|--------|-----|
|odabran pridjev      | 131 | 21|
|odabran ostatak      |4     |0 |
 
|   | glagol | ostalo  |
|-----|-------|-----|
|odabran glagol      | 13 |2 |
|odabran ostatak      | 16     | 7 |

###### Preciznost imenice: 0,94 %-------Opoziv imenice:  0,63 %---------F1 mjera: 0,75
###### Preciznost pridjevi: 0,86 %------- Opoziv pridjevi: 100 %---------F1 mjera: 0,93
###### Preciznost glagoli: 0,87 %-------Opoziv glagoli:  0,65 %---------F1 mjera: 0,74
### 4. Tablice preciznosti i F1 mjere između klasifikatora
##### PRECIZNOST ( zbroj preciznosti po imenici, pridjevu i glagolu, te to sve podijeljeno s 3)
|    | 1. osobina | 2. osobina  |3. osobina |
|------------|----------|-----|-----|
|Maksimalna entropija     | 0.939 |0.897 |0.894|
|Naivni bayes      |0.939     | 0.897 |0.894|

##### F1 mjera ( zbroj mjere po imenici, pridjevu i glagolu, te to sve podijeljeno s 3)
|     | 1. osobina | 2. osobina  |3. osobina| 
|-------|----------|-----|-----|
|Maksimalna entropija     | 0.763 |0.823 |0.806|
|Naivni bayes      |0.763     | 0.823 |0.806|

> Vidimo kako su preciznosti između klasifikatora jednake, isto tako i F1 mjere. Razlika je što je klasifikator maksimalne entropije diskriminativni model, a naivni bayes generativni model. Inače, maksimalna entropija je brža ako imamo 2 klase, no mi u ovom slučaju imamo 3 klase (imenice, pridjevi, glagoli), te su zbog toga jednaki u mjerama, iako ne uvijek nužno. Različiti su u brzini izvedbe. Naivni bayes je brži.
