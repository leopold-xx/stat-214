
# Allowed values by "rule name"
ALLOWED_VALUES = {
    # Binary formats
    "yes_no": {0, 1},
    "yes_no_na": {0, 1, 92},
    "yes_no_unclear": {0, 1, 2},  
    "yes_no_pv": {0, 1, 91},        
    

    # Ordinal formats
    "ordinal_3": {1, 2, 3},
    "ordinal_3_na": {1, 2, 3, 92},
    "ordinal_4": {1, 2, 3, 4},
    "ordinal_4_na": {1, 2, 3, 4, 92},
    "ordinal_5": {1, 2, 3, 4, 5},
    "ordinal_6": {1, 2, 3, 4, 5, 6},

    # Coded categorical formats
    "empl_type": {1, 2, 3, 4, 5},
    "cert_type": {1, 2, 3, 4, 90},
    "inj_mech": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 90},
    "race": {1, 2, 3, 4, 5, 90},
    "ed_disposition": {1, 2, 3, 4, 5, 6, 7, 8, 90},

    # Two-level categories (1/2)
    "two_level": {1, 2},
}


# Column mapped to rule (explicit)
COLUMN_RULES = {
    # Provider / mechanism
    "EmplType": "empl_type",
    "Certification": "cert_type",
    "InjuryMech": "inj_mech",
    "High_impact_InjSev": "ordinal_3",

    # Symptoms / history
    "Amnesia_verb": "yes_no_pv",
    "LOCSeparate": "yes_no_unclear",
    "Seiz": "yes_no",
    "Vomit": "yes_no",
    "Dizzy": "yes_no",
    "ActNorm": "yes_no",
    "HA_verb": "yes_no_pv",
    "Intubated": "yes_no",
    "Paralyzed": "yes_no",
    "Sedated": "yes_no",
    "Drugs": "yes_no",


    # Symptom details (often include 92)
    "LocLen": "ordinal_4_na",
    "SeizOccur": "ordinal_3_na",
    "SeizLen": "ordinal_4_na",
    "VomitNbr": "ordinal_3_na",
    "VomitStart": "ordinal_4_na",
    "VomitLast": "ordinal_3_na",
    "HASeverity": "ordinal_3_na",
    "HAStart": "ordinal_4_na",

    # GCS
    "GCSEye": "ordinal_4",
    "GCSVerbal": "ordinal_5",
    "GCSMotor": "ordinal_6",
    "GCSGroup": "two_level",

    # AMS and subitems
    "AMS": "yes_no",
    "AMSAgitated": "yes_no_na",
    "AMSSleep": "yes_no_na",
    "AMSSlow": "yes_no_na",
    "AMSRepeat": "yes_no_na",
    "AMSOth": "yes_no_na",

    # Skull fracture / exam
    "SFxPalp": "yes_no_unclear",
    "SFxPalpDepress": "yes_no_na",
    "SFxBas": "yes_no",
    "SFxBasHem": "yes_no_na",
    "SFxBasOto": "yes_no_na",
    "SFxBasPer": "yes_no_na",
    "SFxBasRet": "yes_no_na",
    "SFxBasRhi": "yes_no_na",

    # Fontanelle
    "FontBulg": "yes_no",

    # Hematoma
    "Hema": "yes_no",
    "HemaLoc": "ordinal_3_na",
    "HemaSize": "ordinal_3_na",

    # Trauma above clavicles
    "Clav": "yes_no",
    "ClavFace": "yes_no_na",
    "ClavNeck": "yes_no_na",
    "ClavFro": "yes_no_na",
    "ClavOcc": "yes_no_na",
    "ClavPar": "yes_no_na",
    "ClavTem": "yes_no_na",

    # Neuro deficit
    "NeuroD": "yes_no",
    "NeuroDMotor": "yes_no_na",
    "NeuroDSensory": "yes_no_na",
    "NeuroDCranial": "yes_no_na",
    "NeuroDReflex": "yes_no_na",
    "NeuroDOth": "yes_no_na",

    # Other substantial injury
    "OSI": "yes_no",
    "OSIExtremity": "yes_no_na",
    "OSICut": "yes_no_na",
    "OSICspine": "yes_no_na",
    "OSIFlank": "yes_no_na",
    "OSIAbdomen": "yes_no_na",
    "OSIPelvis": "yes_no_na",
    "OSIOth": "yes_no_na",

    # CT process
    "CTForm1": "yes_no",
    "CTSed": "yes_no_na",
    "CTDone": "yes_no",
    "EDCT": "yes_no_na",
    "PosCT": "yes_no_na",

    # Demographics
    "AgeTwoPlus": "two_level",
    "Gender": "two_level",
    "Ethnicity": "two_level",
    "Race": "race",

    # Disposition / outcomes
    "Observed": "yes_no",
    "EDDisposition": "ed_disposition",
    "DeathTBI": "yes_no",
    "HospHead": "yes_no",
    "HospHeadPosCT": "yes_no",
    "Intub24Head": "yes_no",
    "Neurosurgery": "yes_no",
    "PosIntFinal": "yes_no",
}

# for Finging*, Ind*, & CTSed* variables
PREFIX_RULES = {
    "Finding": "yes_no_na",
    "CTSed": "yes_no_na",
    "Ind": "yes_no_na",
}

# Parent-Child Conditional groups

CONDITIONAL_GROUPS = {
    # Symptoms
    "Vomit": ["VomitNbr", "VomitStart", "VomitLast"],
    "LOCSeparate": ["LocLen"],
    "Seiz": ["SeizOccur", "SeizLen"],
    "HA_verb": ["HASeverity", "HAStart"],
    "AMS": ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat", "AMSOth"],

    # clincal findings
    "SFxPalp" :['SFxPalpDepress'],
    "SFxBas": ['SFxBasHem', 'SFxBasOto', "SFxBasPer","SFxBasRet","SFxBasRhi"],
    'Hema': ['HemaLoc', 'HemaSize'],
    'Clav': ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc" ,"ClavPar", "ClavTem"],
    "NeuroD": ['NeuroDMotor', "NeuroDSensory", "NeuroDCranial", "NeuroDReflex", 'NeuroDOth'],
    'OSI': ['OSIExtremity', "OSICut", "OSICspine", "OSIFlank", "OSIAbdomen", "OSIPelvis", "OSIOth"],

    # CT ordering / sedation reasons
    "CTForm1": [
        "IndAge","IndAmnesia","IndAMS","IndClinSFx","IndHA","IndHema","IndLOC","IndMech",
        "IndNeuroD","IndRqstMD","IndRqstParent","IndRqstTrauma","IndSeiz","IndVomit",
        "IndXraySFx","IndOth"
    ],
    "CTSed": ["CTSedAgitate", "CTSedAge", "CTSedRqst", "CTSedOth"],

    # CT actually done / findings
    "CTDone": ["EDCT", "PosCT", "Finding1","Finding2","Finding3","Finding4","Finding5","Finding6",
               "Finding7","Finding8","Finding9","Finding10","Finding11","Finding12","Finding13",
               "Finding14","Finding20","Finding21","Finding22","Finding23"],
    }


STRUCTURAL_DETAIL_COLS = [

    # LOC
    "LocLen",
    "LOCSeparate",

    # Seizure
    "SeizOccur",
    "SeizLen",

    "Amnesia_verb",  

    # Vomiting
    "VomitNbr",
    "VomitStart",
    "VomitLast",

    # Dizzy
    "Dizzy",

    # Headache
    "HA_verb",
    "HASeverity",
    "HAStart",

    # Altered Mental Status
    "AMSAgitated",
    "AMSSleep",
    "AMSSlow",
    "AMSRepeat",
    "AMSOth",

    # Skull Fracture (Palpable)
    "SFxPalpDepress",

    # Basilar Skull Fracture Subtypes
    "SFxBasHem",
    "SFxBasOto",
    "SFxBasPer",
    "SFxBasRet",
    "SFxBasRhi",

    # Hematoma Details
    "HemaLoc",
    "HemaSize",

    # Clavicle / Trauma Above Clavicle Details
    "ClavFace",
    "ClavNeck",
    "ClavFro",
    "ClavOcc",
    "ClavPar",
    "ClavTem",

    # Neurologic Deficit Subtypes
    "NeuroDMotor",
    "NeuroDSensory",
    "NeuroDCranial",
    "NeuroDReflex",
    "NeuroDOth",

    # Other Significant Injury Subtypes
    "OSIExtremity",
    "OSICut",
    "OSICspine",
    "OSIFlank",
    "OSIAbdomen",
    "OSIPelvis",
    "OSIOthr",

    # CT Sedation Details
    "CTSedAgitate",
    "CTSedAge",
    "CTSedRqst",
    "CTSedOth",

    # CT Indication Subtypes 
    "IndAge",
    "IndAmnesia",
    "IndAMS",
    "IndClinSFx",
    "IndHA",
    "IndHema",
    "IndLOC",
    "IndMech",
    "IndNeuroD",
    "IndRqstMD",
    "IndRqstParent",
    "IndRqstTrauma",
    "IndSeiz",
    "IndVomit",
    "IndXraySFx",
    "IndOth"
]

