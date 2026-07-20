"""
5 prompt versions for the rental-listing extraction task, deliberately built
up in layers so you can feel exactly what each addition buys you:

  v1  naive              - just ask, no structure at all
  v2  named fields       - ask for the specific German terms, still free-text
  v3  enforced JSON       - force a parseable output format
  v4  + domain rules      - teach it HOW to compute the tricky fields
  v5  + few-shot example  - show it a fully worked example, robust parsing

Each entry is (name, template). {text} gets filled in with the listing.
"""

PROMPT_VERSIONS = [
    (
        "v1_naive",
        "Extract the rent, deposit, and notice period from this German rental listing.\n\n{text}",
    ),
    (
        "v2_named_fields",
        "Extract the following information from this German rental listing: "
        "Kaltmiete (cold rent), Kaution (deposit), and Kündigungsfrist (notice period).\n\n"
        "Listing:\n{text}",
    ),
    (
        "v3_json_format",
        'Extract information from this German rental listing and return ONLY a valid '
        'JSON object with these exact keys: "kaltmiete", "nebenkosten", "warmmiete", '
        '"kaution", "kuendigungsfrist_monate". All values must be numbers (EUR or months) '
        'or null if not stated. Do not include any text besides the JSON.\n\n'
        "Listing:\n{text}",
    ),
    (
        "v4_domain_rules",
        'Extract structured data from this German rental listing. Return ONLY a valid '
        'JSON object with these exact keys and rules:\n\n'
        '- "kaltmiete": cold rent (Kaltmiete/KM) in EUR, excluding utilities. null if not stated.\n'
        '- "nebenkosten": utility costs (Nebenkosten/NK) in EUR. null if not stated.\n'
        '- "warmmiete": total rent (Warmmiete/WM). Use the stated value if given directly; '
        'otherwise compute kaltmiete + nebenkosten if both are known; otherwise null.\n'
        '- "kaution": deposit in EUR. If given as a multiple of rent (e.g. "3 Nettokaltmieten", '
        '"2 Monatsmieten"), multiply that number by the relevant rent. If the listing says '
        '"keine Kaution", use 0, not null.\n'
        '- "kuendigungsfrist_monate": notice period in months. "gesetzliche Kündigungsfrist" '
        'means the statutory default of 3 months (BGB §573c). null only if truly unmentioned.\n\n'
        'Note: listings may abbreviate Kaltmiete as "KM", Nebenkosten as "NK", Warmmiete as "WM". '
        'German numbers use "." as thousands separator and "," as decimal separator '
        '(e.g. "1.200,00" = 1200.00). Output ONLY the JSON object, no other text.\n\n'
        "Listing:\n{text}",
    ),
    (
        "v5_fewshot_cot",
        'Extract structured data from this German rental listing.\n\n'
        'FIELDS AND RULES:\n'
        '- "kaltmiete": cold rent (Kaltmiete/KM) in EUR. null if not explicitly stated.\n'
        '- "nebenkosten": utility costs (Nebenkosten/NK) in EUR. null if not stated.\n'
        '- "warmmiete": total rent (Warmmiete/WM). Use the stated value if given directly; '
        'otherwise compute kaltmiete + nebenkosten if both known; otherwise null.\n'
        '- "kaution": deposit in EUR. If given as a multiple of rent (e.g. "3 Nettokaltmieten", '
        '"2 Monatsmieten"), multiply that number by the relevant rent. If listing says '
        '"keine Kaution", use 0. null only if truly unmentioned.\n'
        '- "kuendigungsfrist_monate": notice period in months. "gesetzliche Kündigungsfrist" '
        'means the statutory default: 3. null only if truly unmentioned.\n\n'
        'Abbreviations: KM=Kaltmiete, NK=Nebenkosten, WM=Warmmiete. German numbers use "." as '
        'thousands separator and "," as decimal separator (e.g. "1.200,00" = 1200.00).\n\n'
        'EXAMPLE:\n'
        'Listing: "2-Zimmer-Wohnung, Kaltmiete 800 €, NK 150 €. Kaution: 2 Nettokaltmieten. '
        'Es gilt die gesetzliche Kündigungsfrist."\n'
        'Reasoning: kaltmiete=800, nebenkosten=150, warmmiete=800+150=950, kaution=2*800=1600, '
        'notice period is statutory=3.\n'
        'Final JSON: {{"kaltmiete": 800, "nebenkosten": 150, "warmmiete": 950, "kaution": 1600, '
        '"kuendigungsfrist_monate": 3}}\n\n'
        'Now do the same for this listing. You may reason briefly first, but the LAST line of '
        'your response must be ONLY the final JSON object on its own line, with no markdown '
        'fences and no other text on that line.\n\n'
        "Listing:\n{text}",
    ),
]