import pandas as pd
import numpy
import re


def molecular_mass(formula):
    # utilisation du regex vu ici : https://regexp.cheminfo.org/    
    mass = 0.0
    
    parnethesis_pattern = r"\(([^()]+)\)(\d*)"
    matches = re.search(parnethesis_pattern, formula)
    if matches:
        sub_mass = molecular_mass(matches.group(1)) * float(matches.group(2) if matches.group(2) else 1)
        before = formula[:matches.start()]
        after = formula[matches.end():]
        return molecular_mass(before) + sub_mass + molecular_mass(after)

    hydratation_pattern = r"^(.+)\.(\d*)([A-Za-z0-9()]+)$"
    matchhes = re.match(hydratation_pattern,formula)

    # j'en suis la, finir cette partie du code :
    # if matchhes:
    #     sub_mass = molecular_mass(matches.group(3)) * float(matches.group(2) if matches.group(2) else 1)
    #     before = formula[:matches.start()]
    #     after = formula[matches.end():]
    #     return molecular_mass(before) + sub_mass + molecular_mass(after)


    pattern = r"([A-Z][a-z]*)(\d*)"
    matchess = re.findall(pattern, formula)
    for symbol, count in matchess:
        if symbol in element_mass_dict:
            if count != "":
                multiplier = int(count) 
            else:
                multiplier = 1
            mass += element_mass_dict[symbol] * multiplier
    return mass




doc = pd.read_csv("periodic_table.csv")

element_mass_dict = dict(zip(doc["Symbol"], doc["AtomicMass"]))

print("H2O      :", molecular_mass("H2O"))        # 18.015
print("C6H12O6  :", molecular_mass("C6H12O6"))   # 180.156
print("Ca(OH)2  :", molecular_mass("Ca(OH)2"))   # 74.094
print("Fe2(SO4)3:", molecular_mass("Fe2(SO4)3")) # 399.88
print("CuSO4.5H2O:", molecular_mass("CuSO4.5H2O")) 



