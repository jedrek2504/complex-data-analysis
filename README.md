# Analiza danych złożonych z detekcją wyjątków

Zadania laboratoryjne z przedmiotu *Analiza danych złożonych z detekcją wyjątków*
(Politechnika Łódzka, WFTIMS, rok akademicki 2026/2027).

Autor: Jędrzej Kostyk (nr indeksu 264339)

| Zadanie | Temat | Raport |
|---|---|---|
| 1 | Porównanie klasycznych metod detekcji wyjątków | [`report/zadanie1/zadanie1.pdf`](report/zadanie1/zadanie1.pdf) |
| 2 | Wpływ reprezentacji danych (Original, PCA, UMAP, autoenkoder) | [`report/zadanie2/zadanie2.pdf`](report/zadanie2/zadanie2.pdf) |
| 3 | Zespołowa detekcja wyjątków (głosowanie, fuzja ważona, własne warianty DSF i CBF) | [`report/zadanie3/zadanie3.pdf`](report/zadanie3/zadanie3.pdf) |

## Dane

Wyłącznie gotowe zbiory z benchmarków detekcji anomalii. Macierz cech `X` i etykiety
`y` są wczytywane dokładnie tak, jak zostały opublikowane: kod nie generuje, nie usuwa
ani nie podpróbkowuje żadnych obserwacji. Etykiety anomalii pochodzą z benchmarków.
Jedyne operacje na danych to warstwowy podział i standaryzacja dopasowana na części
uczącej.

| Zbiór | Źródło (plik) | $n$ | Cechy | Anomalie | Oryginalne źródło | Zadania |
|---|---|---|---|---|---|---|
| Breastw | ADBench `4_breastw.npz` | 683 | 9 | 239 (35,0 %) | UCI Breast Cancer Wisconsin (Original) | 1 |
| Mammography | ADBench `23_mammography.npz` | 11 183 | 6 | 260 (2,3 %) | Woods i in. (1993), spoza UCI | 1 |
| Wine | ADBench `45_wine.npz` | 129 | 13 | 10 (7,8 %) | UCI Wine | 1 |
| Statlog Heart | DAMI `HeartDisease_withoutdupl_05_v02.arff` | 157 | 13 | 7 (4,5 %) | UCI Statlog (Heart) | 1 |
| Cardio | ADBench `6_cardio.npz` | 1 831 | 21 | 176 (9,6 %) | UCI Cardiotocography | 1, 2, 3 |
| Arrhythmia | ODDS `arrhythmia.mat` | 452 | 274 | 66 (14,6 %) | UCI Arrhythmia | 2, 3 |
| WDBC | ADBench `43_WDBC.npz` | 367 | 30 | 10 (2,7 %) | UCI Breast Cancer Wisconsin (Diagnostic) | 2, 3 |
| Campaign | ADBench `5_campaign.npz` | 41 188 | 62 | 4 640 (11,3 %) | UCI Bank Marketing | 2, 3 |

Źródła:

- **ADBench**: S. Han i in., *ADBench: Anomaly Detection Benchmark*, NeurIPS 2022.
  Pliki z repozytorium `Minqi824/ADBench`, commit
  `3dac8221081e190f157d78e93bfa8867f90d0965`, katalog `adbench/datasets/Classical/`.
  Pobierane automatycznie, bez instalowania pakietu `adbench`.
- **ODDS**: S. Rayana, *ODDS Library*, Stony Brook University, 2016.
  Serwer ODDS blokuje pobieranie automatyczne, więc plik `arrhythmia.mat` trzeba
  pobrać ręcznie ze strony <https://odds.cs.stonybrook.edu/?p=312> i zapisać jako
  `data/raw/odds/arrhythmia.mat`.
- **DAMI**: G. O. Campos i in., *On the evaluation of unsupervised outlier detection:
  measures, datasets, and an empirical study*, Data Mining and Knowledge Discovery
  30(4), 2016. Archiwum `HeartDisease.tar.gz` jest pobierane i rozpakowywane
  automatycznie. Używany plik: `HeartDisease/HeartDisease_withoutdupl_05_v02.arff`,
  czyli wariant bez duplikatów, bez normalizacji, z 5 % anomalii, wersja v02.
  To zbiór Statlog (Heart) z UCI.

Pobrane pliki trafiają do `data/raw/` (katalog nie jest wersjonowany). Ich sumy SHA256
są zapisane w `data/CHECKSUMS.txt`. Przy każdym wczytaniu kod sprawdza sumę, kształt
macierzy i liczbę anomalii (`common/datasets.py`).

## Protokół eksperymentalny

Ten sam protokół obowiązuje we wszystkich zadaniach (`common/protocol.py`):

- Warstwowy podział train / walidacja / test = 50 / 25 / 25 %. Te same ziarna dają
  te same podziały we wszystkich zadaniach.
- Liczba podziałów:
  - 10 (ziarna 0–9) domyślnie;
  - 30 dla zbiorów z co najwyżej 10 anomaliami (Wine, Statlog Heart, WDBC), bo ich
    części walidacyjne i testowe zawierają tylko 2–3 anomalie. Każda część podziału
    zawiera co najmniej jedną anomalię, co sprawdza kod;
  - 5 dla Campaign, ze względu na czas obliczeń.
- Standaryzacja i reprezentacje są dopasowywane tylko na części uczącej.
- Detektory są uczone bez etykiet (scenariusz nienadzorowany).
- Hiperparametry, progi decyzyjne i wagi zespołu są wybierane na części walidacyjnej.
  Część testowa służy wyłącznie do końcowej oceny.
- W zadaniu 3 wszystkie warianty zespołów sprawdzono dodatkowo na nowych podziałach
  (kolejny blok ziaren), nieużywanych przy ich projektowaniu.
- Campaign (41 188 rekordów) jest używany w całości. W zadaniu 2 ma węższą siatkę:
  - UMAP tylko z `min_dist` = 0 i `n_neighbors` ∈ {15, 50};
  - OCSVM bez `nu` = 0,5;
  - miary struktury liczone na 2 000 punktach testowych.

## Najważniejsze wyniki

- **Zadanie 1:** żadna metoda nie dominuje.
  - OCSVM wygrywa na Cardio (PR-AUC 0,76) i Wine (0,89), DBSCAN na Breastw (0,99).
  - LOF i COF zawodzą, gdy anomalie tworzą duże, gęste skupisko (Breastw, 35 % anomalii).
  - Różnice między metodami dotyczą głównie anomalii lokalnych. Globalne wykrywają
    prawie w całości wszystkie metody.
  - Accuracy jest myląca: na Mammography klasyfikator „zawsze normalny” ma wyższą
    Accuracy niż najlepszy detektor.
- **Zadanie 2:** PCA najczęściej poprawia wyniki LOF i OCSVM.
  - WDBC: z 0,71 do 0,92 przy jednej składowej (33 % wariancji).
  - Arrhythmia (274 cechy): poprawa przy 5–10 składowych.
  - UMAP pogarsza detekcję w 10 z 12 par zbiór–detektor, ale UMAP z OCSVM daje najlepszy
    wynik na Cardio (0,84), bo wydziela skupiska anomalii poza masę danych.
  - Na Campaign żadna reprezentacja nie pomaga.
- **Zadanie 3:** najlepsze zespoły (własne warianty DSF i CBF) są średnio tak samo dobre
  jak najlepszy pojedynczy detektor wybrany na walidacji.
  - Wygrywają na Cardio, Arrhythmii i Campaign (+0,01 do +0,04 PR-AUC).
  - Przegrywają na WDBC, gdzie walidacja ma 2–3 anomalie.
  - DSF (zachłanny wybór, wagi skill × różnorodność) sprawdza się przy dużej walidacji,
    CBF (bootstrapowy wybór detektorów kompetentnych, równe wagi) przy małej.
  - Na nowych podziałach CBF miał najwyższą średnią PR-AUC.
- **Pytanie przewodnie:** największy zrealizowany wpływ mają wybór algorytmu i wybór
  reprezentacji, które w praktyce trzeba dobierać jako parę.
  - Wybór metody na walidacji: +0,15 PR-AUC względem domyślnego Isolation Forest
    (zadanie 1).
  - Wybór reprezentacji: +0,12 (zadanie 2).
  - Połączenie detektorów ma najmniejszy wpływ (ok. 0).
  - Ponad tym wszystkim stoi natura danych: na Campaign nic nie podniosło PR-AUC
    powyżej 0,34.

## Struktura repozytorium

```
common/                  wspólny kod: dane, protokół, detektory, COF, metryki, wykresy
task1_classical/         zadanie 1: EDA, siatka hiperparametrów, analiza
task2_representations/   zadanie 2: PCA, UMAP, autoenkoder, eksperymenty, analiza
task3_ensemble/          zadanie 3: normalizacja, głosowanie, fuzja, analiza i synteza
results/taskN/           wyniki (CSV), tabele LaTeX (tables/) i rysunki PDF (figures/)
report/                  raporty LaTeX (XeLaTeX) i skompilowane PDF
```

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Dane: ADBench i DAMI pobierają się automatycznie przy pierwszym użyciu;
# arrhythmia.mat z ODDS trzeba wcześniej zapisać ręcznie w data/raw/odds/

# Zadanie 1
python -m task1_classical.eda
python -m task1_classical.run_grid
python -m task1_classical.run_best
python -m task1_classical.detection_maps
python -m task1_classical.analyze

# Zadanie 2 (równolegle; liczbę procesów ustawia zmienna N_JOBS, domyślnie 3;
# ok. 7 GB RAM wystarcza na 3 procesy)
python -m task2_representations.run
python -m task2_representations.analyze

# Zadanie 3 (wymaga wyników zadania 2)
python -m task3_ensemble.run_base
python -m task3_ensemble.run_ensembles
python -m task3_ensemble.run_base --confirm        # test potwierdzający, nowe ziarna
python -m task3_ensemble.run_ensembles --confirm
python -m task3_ensemble.analyze

# Raporty (XeLaTeX)
cd report/zadanie1 && latexmk -xelatex zadanie1.tex
```

Autoenkoder działa w PyTorch. Domyślnie trening odbywa się na CPU, bo przy kilkuset
obserwacjach jest tam szybszy niż na GPU. Urządzenie można zmienić zmienną
`AE_DEVICE=cuda`.
