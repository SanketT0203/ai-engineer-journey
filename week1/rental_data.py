"""
10 synthetic German rental listings + hand-verified ground truth.

Deliberately includes the messy real-world cases that separate a naive
prompt from a careful one:
  - deposit given only as a MULTIPLE of rent ("3 Nettokaltmieten")
  - warm rent given directly with no cold-rent breakdown
  - "keine Kaution" (explicitly ZERO) vs. a field that's just never mentioned (null)
  - German number formatting (1.200,00 = 1200.00)
  - "gesetzliche Kündigungsfrist" (statutory default = 3 months, BGB §573c)
  - common listing-site abbreviations: KM / NK / WM
"""

FIELDS = ["kaltmiete", "nebenkosten", "warmmiete", "kaution", "kuendigungsfrist_monate"]

LISTINGS = [
    {
        "id": 1,
        "text": (
            "Gemütliche 2-Zimmer-Wohnung in Kreuzberg, 55m². "
            "Kaltmiete: 950 € zzgl. 200 € Nebenkosten. "
            "Kaution: 2.850 € (3 Nettokaltmieten). Kündigungsfrist: 3 Monate."
        ),
        "ground_truth": {"kaltmiete": 950, "nebenkosten": 200, "warmmiete": 1150,
                          "kaution": 2850, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 2,
        "text": (
            "Helle 3-Zimmer-Altbauwohnung, Prenzlauer Berg, 78m². "
            "Miete kalt 1.100 €, Nebenkosten 250 €. "
            "Die Kaution entspricht drei Nettokaltmieten. Kündigungsfrist laut Vertrag: 3 Monate."
        ),
        "ground_truth": {"kaltmiete": 1100, "nebenkosten": 250, "warmmiete": 1350,
                          "kaution": 3300, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 3,
        "text": (
            "1-Zimmer-Apartment nähe Alexanderplatz, 32m². "
            "Warmmiete: 780 € (all-inclusive). Kaution: 1.560 €. "
            "Kündigungsfrist: gesetzlich (3 Monate)."
        ),
        "ground_truth": {"kaltmiete": None, "nebenkosten": None, "warmmiete": 780,
                          "kaution": 1560, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 4,
        "text": (
            "WG-Zimmer in Friedrichshain, 18m² im Rahmen einer 4er-WG. "
            "Miete: 480 € warm. Keine Kaution erforderlich (Zwischenmiete). "
            "Kündigungsfrist: 1 Monat."
        ),
        "ground_truth": {"kaltmiete": None, "nebenkosten": None, "warmmiete": 480,
                          "kaution": 0, "kuendigungsfrist_monate": 1},
    },
    {
        "id": 5,
        "text": (
            "Familienwohnung Charlottenburg, 95m², 4 Zimmer. "
            "Kaltmiete: 1.450,00 EUR, Nebenkosten: 310,50 EUR. "
            "Kaution: 4.350,00 EUR. Kündigungsfrist: 3 Monate zum Quartalsende."
        ),
        "ground_truth": {"kaltmiete": 1450, "nebenkosten": 310.50, "warmmiete": 1760.50,
                          "kaution": 4350, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 6,
        "text": (
            "Modernes Studio in Schöneberg, 40m². "
            "Miete: 890 € kalt zzgl. 150 € NK. Kaution: 2.670 €. "
            "Es gilt die gesetzliche Kündigungsfrist."
        ),
        "ground_truth": {"kaltmiete": 890, "nebenkosten": 150, "warmmiete": 1040,
                          "kaution": 2670, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 7,
        "text": (
            "2-Zimmer-Wohnung Neukölln, 60m². Kaltmiete 780 €, Nebenkosten 190 €. "
            "Kaution: 2 Monatsmieten (Kaltmiete). Kündigungsfrist: 3 Monate."
        ),
        "ground_truth": {"kaltmiete": 780, "nebenkosten": 190, "warmmiete": 970,
                          "kaution": 1560, "kuendigungsfrist_monate": 3},
    },
    {
        "id": 8,
        "text": (
            "Loft-Wohnung in Mitte, 70m², Kaltmiete 1.600 €, Nebenkosten 280 €, "
            "Kaution 4.800 €."
        ),
        "ground_truth": {"kaltmiete": 1600, "nebenkosten": 280, "warmmiete": 1880,
                          "kaution": 4800, "kuendigungsfrist_monate": None},
    },
    {
        "id": 9,
        "text": (
            "Möbliertes Apartment auf Zeit, Wedding, 25m². Warmmiete 650 €, "
            "Kaution 650 € (1 Monatsmiete warm). Kündigungsfrist: 1 Monat (Zwischenmiete)."
        ),
        "ground_truth": {"kaltmiete": None, "nebenkosten": None, "warmmiete": 650,
                          "kaution": 650, "kuendigungsfrist_monate": 1},
    },
    {
        "id": 10,
        "text": (
            "3-Zi-Whg, Steglitz, 85m². KM 1.250 €, NK 275 €, Kaution 2 KM. "
            "Kündigungsfrist: 3 Monate."
        ),
        "ground_truth": {"kaltmiete": 1250, "nebenkosten": 275, "warmmiete": 1525,
                          "kaution": 2500, "kuendigungsfrist_monate": 3},
    },
]