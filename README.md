# Analiza danych złożonych z detekcją wyjątków

Zadania laboratoryjne z przedmiotu *Analiza danych złożonych z detekcją wyjątków*
(Politechnika Łódzka, WFTIMS, rok akademicki 2026/2027).

Autor: Jędrzej Kostyk (nr indeksu 264339)

| Zadanie | Temat | Raport |
|---|---|---|
| 1 | Porównanie klasycznych metod detekcji wyjątków | [`report/zadanie1/zadanie1.pdf`](report/zadanie1/zadanie1.pdf) |
| 2 | Wpływ reprezentacji danych (Original, PCA, UMAP, autoenkoder) | [`report/zadanie2/zadanie2.pdf`](report/zadanie2/zadanie2.pdf) |
| 3 | Zespołowa detekcja wyjątków (głosowanie, fuzja ważona, wariant DSF) | [`report/zadanie3/zadanie3.pdf`](report/zadanie3/zadanie3.pdf) |

## Dane

Zbiory z listy dopuszczonych w kursie, inne niż podane w nim przykłady:

| Zbiór | Cechy | Anomalia | Zadania |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | 30 | guz złośliwy (podpróbkowany do 10 %) | 1, 2, 3 |
| Ionosphere | 33 | odbicie „bad” (podpróbkowane do 10 %) | 1, 2, 3 |
| Parkinsons | 22 | nagranie osoby zdrowej (klasa mniejszościowa) | 1, 2, 3 |
| Glass Identification | 9 | szkło stołowe, klasa 6 (4,2 %) | 1 |
| Syntetyczny gaussowski | 30 | anomalie globalne, lokalne i podprzestrzenne (10 %) | 1, 2, 3 |

Breast Cancer pochodzi z `scikit-learn`. Pozostałe zbiory rzeczywiste są
pobierane automatycznie z repozytorium UCI do `data/raw/` (katalog nie jest
wersjonowany). Zbiór syntetyczny jest generowany deterministycznie.

## Protokół eksperymentalny

Ten sam protokół obowiązuje we wszystkich zadaniach (`common/protocol.py`):

- 10 ziaren losowych, a dla każdego warstwowy podział train / walidacja / test = 50 / 25 / 25 %.
  Te same ziarna dają te same podziały we wszystkich zadaniach.
- Standaryzacja i reprezentacje są dopasowywane tylko na części uczącej.
- Detektory są uczone bez etykiet (scenariusz nienadzorowany).
- Hiperparametry, progi decyzyjne i wagi zespołu są wybierane na części walidacyjnej.
  Część testowa służy wyłącznie do końcowej oceny.
- W zadaniu 3 wszystkie warianty zespołów sprawdzono dodatkowo na 10 nowych podziałach
  (ziarna 10–19), nieużywanych przy ich projektowaniu.

## Najważniejsze wyniki

- **Zadanie 1:** żadna metoda nie dominuje. LOF jako jedyny wykrywa wyjątki lokalne
  (Recall 0,66 wobec 0,00 dla Isolation Forest i DBSCAN), a OCSVM z jądrem RBF
  i DBSCAN najlepiej radzą sobie z anomaliami globalnymi (Ionosphere, PR-AUC 0,90).
- **Zadanie 2:** PCA poprawia wyniki w 6 z 12 par zbiór–detektor (zawsze dla
  Isolation Forest), a jej błąd rekonstrukcji jest najlepszym detektorem na Ionosphere
  (0,93). UMAP pogarsza detekcję, bo wyrównuje lokalną gęstość. Przy 91 % zachowanej
  wariancji PCA traci niemal całą informację o anomaliach podprzestrzennych.
- **Zadanie 3:** zespół nie przewyższa średnio najlepszego detektora wybranego na
  walidacji. Wygrywa tylko tam, gdzie detektory są porównywalnie dobre (Ionosphere).
  Własny wariant CBF (bootstrapowy wybór detektorów kompetentnych) był najlepszym
  wariantem zespołowym w obu seriach podziałów.
- **Pytanie przewodnie:** największy zrealizowany wpływ na jakość detekcji miał wybór
  algorytmu (+0,18 PR-AUC względem domyślnego Isolation Forest). Reprezentacja ma
  największy rozrzut, ale głównie w dół, a połączenie detektorów ma najmniejszy wpływ.

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

# Zadanie 1
python -m task1_classical.eda
python -m task1_classical.run_grid
python -m task1_classical.run_best
python -m task1_classical.decision_regions
python -m task1_classical.analyze

# Zadanie 2 (równolegle; liczbę procesów ustawia zmienna N_JOBS, domyślnie 3)
python -m task2_representations.run
python -m task2_representations.analyze

# Zadanie 3 (wymaga wyników zadania 2)
python -m task3_ensemble.run_base
python -m task3_ensemble.run_ensembles
python -m task3_ensemble.run_base --confirm        # test potwierdzający, ziarna 10-19
python -m task3_ensemble.run_ensembles --confirm
python -m task3_ensemble.analyze

# Raporty (XeLaTeX)
cd report/zadanie1 && latexmk -xelatex zadanie1.tex
```

Autoenkoder działa w PyTorch. Domyślnie trening odbywa się na CPU, bo przy kilkuset
obserwacjach jest tam szybszy niż na GPU. Urządzenie można zmienić zmienną
`AE_DEVICE=cuda`.
