# FiremanProblemEAProject
Repository for the project for the Evolutionary Algorithms course, 2024/2025 edition


Instancja problemu:
- $G = \langle V, E\rangle$ - graf
- $S\subseteq V$ - Początek pożarów
- $N$ - ilość strażaków do rozstawienia


### Algo 1, SGA
Chromosom:
- $\{0,1\}^{|V|}$

Krzyżowanie:
1. Wybierz rodziców (Jakoś)
2. Wylosuj pozycję cięcia
3. Zamień suffix rodziców 

Mutacja 1:
Wylosuj dwa indeksy, zamień ich wartości, napraw osobnika

Mutacja 2:
Wylosuj indeks początkowy, wylosuj długość, wytnij fragment i wylosuj zastępcę
o tej samej ilości strażaków

Mutacja 3:
Parametry:
    - $k$ - ilość strażaków,
    - $m$ - długość spaceru
Wylosuj $k$ strażaków, puść ich na losowy spacer długości $m$

Metoda naprawiania:
- Wylosuj jedynkę, zamień na zero, powtarzaj do momentu, gdy wektor jest poprawny

### Algo 2, (CMA-ES like)
Chromosom:
- $(0,1)^{|V|}$

Kolokwialnie wektor prawdopodobieństw

Metoda naprawiania:
- Normalizuj wyniki.


TODO: Zaprojektuj sposób na tworzenie nowej populacji
