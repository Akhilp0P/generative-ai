"""Domain vocabulary for ESG keyword extraction."""

ESG_TERMS = {
    "Environmental": [
        "scope 1 emissions", "scope 2 emissions", "scope 3 emissions", "scope 1", "scope 2", "scope 3",
        "greenhouse gas emissions", "greenhouse gas", "ghg emissions", "carbon emissions", "carbon intensity",
        "climate change", "climate risk", "net zero", "decarbonization", "renewable energy",
        "renewable electricity", "energy efficiency", "energy consumption", "water consumption", "water withdrawal",
        "water intensity", "wastewater", "effluent", "waste management", "hazardous waste", "non-hazardous waste",
        "recycling", "circular economy", "biodiversity", "ecosystem", "deforestation", "land use", "air emissions", "pollution"
    ],
    "Social": [
        "occupational health and safety", "occupational health", "health and safety", "employee health", "employee safety",
        "lost time injury", "ltifr", "employee engagement", "employee turnover", "diversity and inclusion", "diversity",
        "inclusion", "gender diversity", "human rights", "labour rights", "child labor", "forced labor",
        "training and development", "employee training", "skills development", "community development", "community investment",
        "social impact", "customer privacy", "data privacy", "customer safety", "product safety", "responsible sourcing",
        "supply chain", "supplier diversity"
    ],
    "Governance": [
        "board independence", "independent directors", "board diversity", "audit committee", "risk committee",
        "nomination committee", "remuneration committee", "corporate governance", "business ethics", "code of conduct",
        "anti corruption", "anti-corruption", "anti bribery", "anti-bribery", "whistleblower", "whistleblowing", "speak up",
        "risk management", "enterprise risk management", "regulatory compliance", "compliance", "executive remuneration",
        "executive compensation", "related party transactions", "related-party transactions", "data governance",
        "cybersecurity", "information security", "tax governance"
    ],
}

FRAMEWORK_MAP = {
    "scope 1 emissions": ["GRI", "SASB", "BRSR", "ISSB"],
    "scope 2 emissions": ["GRI", "SASB", "BRSR", "ISSB"],
    "scope 3 emissions": ["GRI", "SASB", "BRSR", "ISSB"],
    "greenhouse gas emissions": ["GRI", "SASB", "BRSR", "ISSB"],
    "water consumption": ["GRI", "SASB", "BRSR"],
    "waste management": ["GRI", "SASB", "BRSR"],
    "biodiversity": ["GRI", "BRSR"],
    "health and safety": ["GRI", "SASB", "BRSR"],
    "human rights": ["GRI", "SASB", "BRSR"],
    "diversity": ["GRI", "SASB", "BRSR", "ISSB"],
    "board independence": ["GRI", "BRSR", "ISSB"],
    "anti corruption": ["GRI", "BRSR"],
    "whistleblower": ["GRI", "BRSR"],
    "risk management": ["BRSR", "ISSB"],
}

def normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())

def classify_esg(keyword: str) -> str:
    text = normalize_text(keyword)
    for category, terms in ESG_TERMS.items():
        if any(term in text for term in terms):
            return category
    return "Unclassified"

def framework_mapping(keyword: str) -> list[str]:
    text = normalize_text(keyword)
    matches = set()
    for term, frameworks in FRAMEWORK_MAP.items():
        if term in text:
            matches.update(frameworks)
    return sorted(matches)
