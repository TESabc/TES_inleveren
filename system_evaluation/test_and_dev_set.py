import json


development_set = [
    {
        'question': 'Hoeveel gevangenissen zijn er in Rotterdam?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?geb) AS ?aantal)
        WHERE {
            ?vbo a sor:Verblijfsobject;
                sor:hoofdadres ?na;
                sor:maaktDeelUitVan ?geb.


            ?gebz sor:hoortBij ?vbo;
                kad:gebouwtype kad-con:gevangenis.

            ?woonplaats a sor:Woonplaats;
                skos:prefLabel "Rotterdam"@nl;
                ^sor:ligtIn ?openbareruimte.
            ?openbareruimte a sor:OpenbareRuimte;
                ^sor:ligtAan ?na.
            ?na a sor:Nummeraanduiding.

        }
        """,
        'generalization_level': 'iid',
        'granularity': 'Woonplaats',
        'location': 'Rotterdam',
        'novel_functions': []
    },

    {
        'question': 'Wat is het oudste ziekenhuis in provincie gelderland?',
        'sparql': """
        SELECT DISTINCT ?geb
        WHERE {
          ?vbo a sor:Verblijfsobject;
               sor:maaktDeelUitVan ?geb.
          ?geb sor:oorspronkelijkBouwjaar ?bo.


          ?gebz sor:hoortBij ?vbo;
               kad:gebouwtype kad-con:ziekenhuis.
          ?gemeente geo:sfWithin provincie:0025.
          ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
        }
        ORDER BY ASC (?bo)
        LIMIT 1
        """,
        'generalization_level': 'iid',
        'granularity': 'Provincie',
        'location': '0025',
        'novel_functions': []
    },

    {
        'question': 'geef mij het grootste perceel in gemeente eindhoven behorend tot een kerk',
        'sparql': """
        SELECT DISTINCT ?perceel WHERE {
            ?perceel a sor:Perceel ;
                sor:hoortBij/^sor:hoofdadres/^sor:hoortBij ?gebouwzone ;
                sor:oppervlakte ?oppervlakte .
            ?gebouwzone kad:gebouwtype kad-con:kerk .
            ?gemeente sdo0:identifier "GM0772" ;
                ^owl:sameAs/^geo:sfWithin ?perceel .
        } ORDER BY DESC(?oppervlakte) LIMIT 1
        """,
        'generalization_level': 'iid',
        'granularity': 'Gemeente',
        'location': '0772',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel gebouwen zijn er in Deventer?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?geb) as ?aantal)
        WHERE {
          ?vbo a sor:Verblijfsobject;
               sor:hoofdadres ?na;
               sor:maaktDeelUitVan ?geb.

          ?geb a sor:Gebouw.

          ?woonplaats a sor:Woonplaats;
                      skos:prefLabel "Deventer"@nl;
                      ^sor:ligtIn ?openbareruimte.
          ?openbareruimte a sor:OpenbareRuimte;
                           ^sor:ligtAan ?na.
        }


        """,
        'generalization_level': 'compositional',
        'granularity': 'Woonplaats',
        'location': 'Deventer',
        'novel_functions': []
    },

    {
        'question': 'Ik wil het grootste perceel in Nederland en de bijbehorende perceeloppervlakte',
        'sparql': """
        SELECT DISTINCT ?per ?po
        WHERE {
            ?per a sor:Perceel;
                 sor:oppervlakte ?po.

        }
        ORDER BY DESC (?po)
        LIMIT 1
        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Geef mij het oudste gebouw in provincie Overijssel en het bijbehorende bouwjaar',
        'sparql': """
        SELECT DISTINCT ?geb ?bo
        WHERE {
          ?vbo a sor:Verblijfsobject;
               sor:maaktDeelUitVan ?geb.

          ?geb a sor:Gebouw;
            sor:oorspronkelijkBouwjaar ?bo.

          ?gemeente geo:sfWithin provincie:0023.
          ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
        }
        ORDER BY ASC (?bo)
        LIMIT 1

        """,
        'generalization_level': 'compositional',
        'granularity': 'Provincie',
        'location': '0023',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel ligplaatsen zijn er in Zutphen?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?ligplaats) as ?ligplaatsAantal)


        WHERE {
          ?ligplaats a kad:Ligplaats;
               sor:hoofdadres ?na.

          ?woonplaats a sor:Woonplaats;
                      skos:prefLabel "Zutphen"@nl;
                      ^sor:ligtIn ?openbareruimte.
          ?openbareruimte a sor:OpenbareRuimte;
                           ^sor:ligtAan ?na.
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Zutphen',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 4
    },

    {
        'question': 'Geef mij alle woonplaatsen met meer dan 200 ligplaatsen daarin. Geef me de woonplaatsen en het aantal ligplaatsen.',
        'sparql': """
        SELECT DISTINCT ?woonplaats (COUNT(DISTINCT ?ligplaats) as ?ligplaatsAantal)

        WHERE {
          ?ligplaats a kad:Ligplaats;
               sor:hoofdadres ?na.

          ?woonplaats a sor:Woonplaats;
                      ^sor:ligtIn ?openbareruimte.
          ?openbareruimte a sor:OpenbareRuimte;
                           ^sor:ligtAan ?na.
        }
        GROUP BY ?woonplaats
        HAVING (COUNT(DISTINCT ?ligplaats) > 200)


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'HAVING', 'MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Wat is het gemiddelde aantal kad ligplaatsen in een sor woonplaats?',
        'sparql': """
        SELECT (AVG(?count) as ?average) WHERE {
          SELECT (COUNT(DISTINCT ?ligplaats) as ?count) ?woonplaats WHERE {
            ?ligplaats a kad:Ligplaats ;
                       sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats .
          } GROUP BY ?woonplaats
        }


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'AVERAGE', 'SUBQUERY'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Hoeveel sor verblijfsobjecten hebben zowel een hoofdadres als een nevenadres in Nederland?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?vbo) as ?vboMetZowelHoofdAlsNevenAdres)

        WHERE {
          ?vbo a sor:Verblijfsobject;
               sor:hoofdadres ?hoofdadres;
               sor:nevenadres ?nevenadres.
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 2
    },

    {
        'question': 'Geef mij alle sor woonplaatsen waarvan de preflabel met Ap begint. Geef mij alleen de woonsplaats zonder preflabel.',
        'sparql': """

        SELECT DISTINCT ?woonplaats WHERE {
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel ?prefLabel .
            FILTER (STRSTARTS(?prefLabel, "Ap"))
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij de woonplaats met de meeste openbare ruimtes daarin die een preflabel hebben beginnend met A. Geef mij de woonplaats en het aantal openbare ruimtes.',
        'sparql': """
        SELECT ?woonplaats (COUNT(DISTINCT ?openbareRuimte) AS ?aantalOpenbareRuimtes) WHERE {
            ?openbareRuimte a sor:OpenbareRuimte ;
                sor:ligtIn ?woonplaats ;
                skos:prefLabel ?prefLabel .
            ?woonplaats a sor:Woonplaats .
            FILTER (STRSTARTS(LCASE(STR(?prefLabel)), "a"))
        } 
        GROUP BY ?woonplaats
        ORDER BY DESC(?aantalOpenbareRuimtes)
        LIMIT 1

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING', 'GROUP BY'],
        'new_schema_item': False,
        'number_of_relations': 2
    },

    {
        'question': 'Geef mij de gebouwzone en de hoogte van de gebouwzone met de hoogste hoogte.',
        'sparql': """
        SELECT DISTINCT ?gebouwzone ?hoogteniveau 
        WHERE {
        ?gebouwzone kad:hoogte ?hoogteniveau.
        }
        ORDER BY DESC(?hoogteniveau)
        LIMIT 1


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij de gebouwzone die de laagste hoogte heeft. Geef mij de gebouwzone en het bijbehorende gebouwtype.',
        'sparql': """
        SELECT ?gebouwzone ?gebouwtype WHERE {
            ?gebouwzone a sor:Gebouwzone ;
                        kad:hoogte ?hoogte ;
                        kad:gebouwtype ?gebouwtype .
        } ORDER BY ASC(?hoogte) LIMIT 1


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 2
    },

    {
        'question': 'Geef mij de sor openbare ruimte met de meeste sor nummeraanduidingen in Amsterdam. Geef mij de openbare ruimte en het aantal nummeraanduidingen.',
        'sparql': """
        SELECT ?openbareruimte (COUNT(DISTINCT ?nummeraanduiding) AS ?aantalNummeraanduidingen)
        WHERE {
            ?nummeraanduiding a sor:Nummeraanduiding ;
                sor:ligtAan ?openbareruimte .
            ?openbareruimte a sor:OpenbareRuimte ;
                sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Amsterdam"@nl .
        }
        GROUP BY ?openbareruimte
        ORDER BY DESC(?aantalNummeraanduidingen)
        LIMIT 1


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Amsterdam',
        'novel_functions': ['MULTIPLE VARIABLES', 'GROUP BY'],
        'new_schema_item': False,
        'number_of_relations': 3
    },

    {
        'question': 'Wat is het gemiddelde bouwjaar van gebouwen in Apeldoorn?',
        'sparql': """
    SELECT (AVG(YEAR(?bouwjaar)) as ?gemBouwjaar) WHERE {
        ?vbo a sor:Verblijfsobject ;
            sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats ;
            sor:maaktDeelUitVan ?gebouw .
        ?gebouw a sor:Gebouw ;
            sor:oorspronkelijkBouwjaar ?bouwjaar .
        ?woonplaats a sor:Woonplaats ;
            skos:prefLabel "Apeldoorn"@nl .
    }



    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Apeldoorn',
        'novel_functions': ['AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 6
    }

]

test_set = [
    {
        'question': 'Hoeveel gevangenissen zijn er in Vught?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?geb) AS ?aantal)
        WHERE {
            ?vbo a sor:Verblijfsobject;
                sor:hoofdadres ?na;
                sor:maaktDeelUitVan ?geb.
            ?geb sor:oorspronkelijkBouwjaar ?bo.


            ?gebz sor:hoortBij ?vbo;
                kad:gebouwtype kad-con:gevangenis.

            ?woonplaats a sor:Woonplaats;
                skos:prefLabel "Vught"@nl;
                ^sor:ligtIn ?openbareruimte.
            ?openbareruimte a sor:OpenbareRuimte;
                ^sor:ligtAan ?na.
            ?na a sor:Nummeraanduiding.

        }
        """,
        'generalization_level': 'iid',
        'granularity': 'Woonplaats',
        'location': 'Vught',
        'novel_functions': []
    },

    {
        'question': 'Wat is het oudste ziekenhuis in provincie Utrecht?',
        'sparql': """
        SELECT DISTINCT ?geb
        WHERE { 
            ?vbo a sor:Verblijfsobject;
                 sor:maaktDeelUitVan ?geb.
            ?geb a sor:Gebouw;
                 sor:oorspronkelijkBouwjaar ?bo.
            ?gebz sor:hoortBij ?vbo;
                  kad:gebouwtype kad-con:ziekenhuis.
            ?gemeente geo:sfWithin provincie:0026.
            ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
        }
        ORDER BY ASC (?bo)
        LIMIT 1

    """,
        'generalization_level': 'iid',
        'granularity': 'Provincie',
        'location': '0026',
        'novel_functions': []
    },

    {
        'question': 'geef mij het grootste perceel in gemeente deventer behorend tot een kerk',
        'sparql': """
        SELECT DISTINCT ?per
        WHERE { 
            ?vbo a sor:Verblijfsobject;
                 sor:hoofdadres ?na;
                 sor:maaktDeelUitVan ?geb.
            ?per a sor:Perceel;
                 sor:hoortBij ?na;
                 sor:oppervlakte ?po.
            ?gebz sor:hoortBij ?vbo;
                  kad:gebouwtype kad-con:kerk.
            ?gemeente sdo0:identifier "GM0150".
            ?gemeente ^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
        }
        ORDER BY DESC (?po)
        LIMIT 1  
    """,
        'generalization_level': 'iid',
        'granularity': 'Gemeente',
        'location': '0150',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel gebouwen zijn tussen bouwjaar 1900 en 1950 in Rotterdam gebouwd?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?geb) as ?aantal)
        WHERE { 
            ?vbo a sor:Verblijfsobject;
                 sor:hoofdadres ?na;
                 sor:maaktDeelUitVan ?geb.
            ?geb a sor:Gebouw;
                 sor:oorspronkelijkBouwjaar ?bo.
            ?woonplaats a sor:Woonplaats;
                        skos:prefLabel "Rotterdam"@nl;
                        ^sor:ligtIn ?openbareruimte.
            ?openbareruimte a sor:OpenbareRuimte;
                             ^sor:ligtAan ?na.
            FILTER (str(?bo) > '1900')
            FILTER (str(?bo) < '1950')
        }
        LIMIT 9999 
    """,
        'generalization_level': 'iid',
        'granularity': 'Woonplaats',
        'location': 'Rotterdam',
        'novel_functions': []
    },

    {
        'question': 'hoeveel kerken staan in gemeente Apeldoorn?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?geb) as ?aantal)
        WHERE { 
            ?vbo a sor:Verblijfsobject;
                 sor:maaktDeelUitVan ?geb.
            ?geb a sor:Gebouw.
            ?gebz sor:hoortBij ?vbo;
                  kad:gebouwtype kad-con:kerk.
            ?gemeente sdo0:identifier "GM0200".
            ?gemeente ^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
        }
        LIMIT 9999

    """,
        'generalization_level': 'iid',
        'granularity': 'Gemeente',
        'location': '0200',
        'novel_functions': []
    },

    {
        'question': 'Wat is het oudste politiebureau in provincie Gelderland?',
        'sparql': """
        SELECT DISTINCT ?geb 
        WHERE { 
            ?vbo a sor:Verblijfsobject; 
                 sor:maaktDeelUitVan ?geb. 
            ?geb a sor:Gebouw; 
                 sor:oorspronkelijkBouwjaar ?bo. 
            ?gebz sor:hoortBij ?vbo; 
                  kad:gebouwtype kad-con:politiebureau. 
            ?gemeente geo:sfWithin provincie:0025. 
            ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb. 
        } 
        ORDER BY ASC(?bo) 
        LIMIT 1
    """,
        'generalization_level': 'iid',
        'granularity': 'Provincie',
        'location': '0025',
        'novel_functions': []
    },

    {
        'question': 'ik wil alle percelen en de geometrie daarvan in lettele',
        'sparql': """
        SELECT DISTINCT ?per ?geo_wgs84 
        WHERE { 
            ?per a sor:Perceel; 
                 sor:hoortBij ?na; 
                 geo:hasGeometry/geo:asWKT ?geo_wgs84. 
            ?woonplaats a sor:Woonplaats; 
                        skos:prefLabel "Lettele"@nl; 
                        ^sor:ligtIn ?openbareruimte. 
            ?openbareruimte a sor:OpenbareRuimte; 
                            ^sor:ligtAan ?na. 
        } 
        LIMIT 9999

    """,
        'generalization_level': 'iid',
        'granularity': 'Woonplaats',
        'location': 'Lettele',
        'novel_functions': []
    },

    {
        'question': 'Geef mij het oudste zwembad in nederland',
        'sparql': """
        SELECT DISTINCT ?geb 
        WHERE { 
            ?vbo a sor:Verblijfsobject; 
                 sor:maaktDeelUitVan ?geb. 
            ?geb a sor:Gebouw; 
                 sor:oorspronkelijkBouwjaar ?bo. 
            ?gebz sor:hoortBij ?vbo; 
                  kad:gebouwtype kad-con:zwembad. 
        } 
        ORDER BY ASC (?bo) 
        LIMIT 1

        """,
        'generalization_level': 'iid',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Geef mij de oudste school in provincie drenthe',
        'sparql': """
        SELECT DISTINCT ?geb 
        WHERE { 
            ?vbo a sor:Verblijfsobject; 
                 sor:maaktDeelUitVan ?geb. 
            ?geb a sor:Gebouw; 
                 sor:oorspronkelijkBouwjaar ?bo. 
            ?gebz sor:hoortBij ?vbo; 
                  kad:gebouwtype kad-con:school. 
            ?gemeente geo:sfWithin provincie:0022. 
            ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb. 
        } 
        ORDER BY ASC (?bo) 
        LIMIT 1


        """,
        'generalization_level': 'iid',
        'granularity': 'Provincie',
        'location': '0022',
        'novel_functions': []
    },

    {
        'question': 'Geef mij de nieuwste brandweerkazerne in gemeente eindhoven',
        'sparql': """
        SELECT DISTINCT ?geb 
        WHERE { 
            ?vbo a sor:Verblijfsobject; 
                 sor:maaktDeelUitVan ?geb. 
            ?geb a sor:Gebouw; 
                 sor:oorspronkelijkBouwjaar ?bo. 
            ?gebz sor:hoortBij ?vbo; 
                  kad:gebouwtype kad-con:brandweerkazerne. 
            ?gemeente sdo0:identifier "GM0772". 
            ?gemeente ^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb. 
        } 
        ORDER BY DESC (?bo) 
        LIMIT 1



        """,
        'generalization_level': 'iid',
        'granularity': 'Gemeente',
        'location': '0772',
        'novel_functions': []
    },

    # COMPOSITIONAL QUESTIONS START HERE

    {
        'question': 'Hoeveel openbare ruimtes zijn er in Apeldoorn?',
        'sparql': """
        SELECT DISTINCT (COUNT(DISTINCT ?openbareruimte) as ?aantal)
        WHERE {
          ?openbareruimte a sor:OpenbareRuimte. 

          ?woonplaats a sor:Woonplaats; 
                      skos:prefLabel "Apeldoorn"@nl;
                      ^sor:ligtIn ?openbareruimte.  
        }
        LIMIT 9999

    """,
        'generalization_level': 'compositional',
        'granularity': 'Woonplaats',
        'location': 'Apeldoorn',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel gemeentes zijn er in provincie Zeeland?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?gemeente) as ?aantal)
        WHERE {    
            ?gemeente geo:sfWithin provincie:0029;
                      a sor:Gemeente.  
        }
        LIMIT 9999


        """,
        'generalization_level': 'compositional',
        'granularity': 'Provincie',
        'location': 'Zeeland',
        'novel_functions': []
    },

    {
        'question': 'Geef mij het perceel welke het kleinste perceel oppervlakte heeft in Nederland. Geef mij alleen het perceel.',
        'sparql': """
        SELECT ?perceel WHERE {
            ?perceel a sor:Perceel ;
                sor:oppervlakte ?oppervlakte .
        } ORDER BY ASC(?oppervlakte) LIMIT 1


    """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Ik wil het kleinste perceel in Nederland en de bijbehorende perceeloppervlakte',
        'sparql': """
        SELECT DISTINCT ?per ?po
        WHERE {
            ?per a sor:Perceel;
                 sor:oppervlakte ?po.

        }
        ORDER BY ASC (?po)
        LIMIT 1



    """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Geef mij alle verblijfsobjecten met celfunctie in Almere',
        'sparql': """
        SELECT DISTINCT ?verblijfsobject 
        WHERE { 
            ?verblijfsobject a sor:Verblijfsobject ;
            sor:gebruiksdoel sor-con:celfunctie ;
            sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
            skos:prefLabel "Almere"@nl .
        }




    """,
        'generalization_level': 'compositional',
        'granularity': 'Woonplaats',
        'location': 'Almere',
        'novel_functions': []
    },

    {
        'question': 'welke gebruiksdoelen kan een verblijfsobject hebben?',
        'sparql': """
        SELECT DISTINCT ?gebruiksdoel WHERE {
          ?vbo a sor:Verblijfsobject ;
               sor:gebruiksdoel ?gebruiksdoel .
        }





        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel gebouwzones zijn er van het type kerk?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?gebouwzone) AS ?count) WHERE {
            ?gebouwzone a sor:Gebouwzone ;
                kad:gebouwtype kad-con:kerk .
        }






        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel verblijfsobjecten vallen onder een gebouwzone met gebouwtype zwembad?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?vbo) AS ?count) WHERE {
          ?gebouwzone a sor:Gebouwzone ;
                      kad:gebouwtype kad-con:zwembad ;
                      sor:hoortBij ?vbo .
          ?vbo a sor:Verblijfsobject .
        }
        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Hoeveel verblijfsobjecten vallen onder een zwembad?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?vbo) AS ?count) WHERE {
          ?vbo a sor:Verblijfsobject .
          ?gebz a sor:Gebouwzone ;
                sor:hoortBij ?vbo ;
                kad:gebouwtype kad-con:zwembad .
        }

        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Wat voor gebouwtypes kan een gebouwzone hebben?',
        'sparql': """
        SELECT DISTINCT ?gebouwtype WHERE {
          ?gebouwzone a sor:Gebouwzone ;
                      kad:gebouwtype ?gebouwtype .
        }
        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    {
        'question': 'Beschouw alle verblijfsobjecten en perceel combinaties. Hoeveel verblijfsobjecten hebben een kleinere oppervlakte vergeleken met het perceel?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?vbo) AS ?count) WHERE {
            ?vbo a sor:Verblijfsobject ;
                 sor:oppervlakte ?vbo_opp ;
                 sor:hoofdadres ?na .
            ?per a sor:Perceel ;
                 sor:hoortBij ?na ;
                 sor:oppervlakte ?per_opp .
            FILTER (?vbo_opp < ?per_opp)
        }

        """,
        'generalization_level': 'compositional',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': []
    },

    # NOW WE DO THE ZERO-SHOT QUESTIONS

    {
        'question': 'Welke woonplaats heeft de meeste ligplaatsen? Geef mij de woonplaats en het aantal ligplaatsen dat daar inzitten.',
        'sparql': """
        SELECT ?woonplaats (COUNT(DISTINCT ?ligplaats) AS ?aantalLigplaatsen) WHERE {
            ?ligplaats a kad:Ligplaats ;
                sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats .
        } GROUP BY ?woonplaats ORDER BY DESC(?aantalLigplaatsen) LIMIT 1

    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES', 'GROUP BY'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Geef mij alle woonplaatsen met meer dan 100 ligplaatsen daarin. Geef me de woonplaatsen en het aantal ligplaatsen.',
        'sparql': """
        SELECT DISTINCT ?woonplaats (COUNT(DISTINCT ?ligplaats) as ?ligplaatsAantal)

        WHERE {
          ?ligplaats a kad:Ligplaats;
               sor:hoofdadres ?na.

          ?woonplaats a sor:Woonplaats;
                      ^sor:ligtIn ?openbareruimte.
          ?openbareruimte a sor:OpenbareRuimte;
                           ^sor:ligtAan ?na.
        }
        GROUP BY ?woonplaats
        HAVING (COUNT(DISTINCT ?ligplaats) > 100)


    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES', 'GROUP BY', 'HAVING'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Hoeveel standplaatsen zijn er in Kampen?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?standplaats) AS ?aantalStandplaatsen) WHERE {
            ?standplaats a kad:Standplaats ;
                sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Kampen"@nl .
        }


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Kampen',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 4
    },

    {
        'question': 'Geef mij alle sor woonplaatsen die meer dan 50 kad standplaatsen bevatten. Geef mij de woonplaatsen en het aantal standplaatsen.',
        'sparql': """
        SELECT ?woonplaats (COUNT(DISTINCT ?standplaats) AS ?aantalStandplaatsen)
        WHERE {
            ?standplaats a kad:Standplaats ;
                sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats .
        }
        GROUP BY ?woonplaats
        HAVING (COUNT(DISTINCT ?standplaats) > 50)


        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'HAVING', 'MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Geef mij het perceel, perceelnummer, de sectie, en de oppervlakte van het perceel met de grootste oppervlakte in Nederland.',
        'sparql': """
        SELECT DISTINCT ?per ?perceelOppervlakte ?perceelnummer ?sectie
        WHERE {
            ?per a sor:Perceel;
                 sor:oppervlakte ?perceelOppervlakte;
                 sor:perceelnummer ?perceelnummer;
                 sor:sectie ?sectie.


        }
        ORDER BY DESC (?po)
        LIMIT 1


    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Geef mij de 3 gebouwzones en de hoogtes daarvan, welke de hoogste hoogte hebben.',
        'sparql': """
        SELECT ?gebouwzone ?hoogte
        WHERE {
          ?gebouwzone a sor:Gebouwzone ;
                      kad:hoogte ?hoogte .
        }
        ORDER BY DESC(?hoogte)
        LIMIT 3
        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij alle kad overheidsorganisaties en hun preflabel.',
        'sparql': """
        SELECT DISTINCT ?org ?label
        WHERE {
            ?org a kad:Overheidsorganisatie;
                skos:prefLabel ?label.
        }




    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij het gebouw met de meeste verblijfsobjecten in Amsterdam? Geef mij alleen het gebouw.',
        'sparql': """
        SELECT DISTINCT ?geb
        WHERE {
          ?vbo a sor:Verblijfsobject;
               sor:hoofdadres ?na;
               sor:maaktDeelUitVan ?geb.

          ?geb a sor:Gebouw.

          ?woonplaats a sor:Woonplaats;
                      skos:prefLabel "Amsterdam"@nl;
                      ^sor:ligtIn ?openbareruimte.
          ?openbareruimte a sor:OpenbareRuimte;
                           ^sor:ligtAan ?na.
        }
        GROUP BY ?geb
        ORDER BY DESC(COUNT(?vbo))
        LIMIT 1



        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Amsterdam',
        'novel_functions': ['MULTIPLE VARIABLES', 'GROUP BY'],
        'new_schema_item': False,
        'number_of_relations': 5
    },

    {
        'question': 'Geef mij alle openbare ruimtes die meer dan 200 verblijfsobjecten bevatten in Dordrecht. Geef mij de openbare ruimtes en de preflabels.',
        'sparql': """
        SELECT ?openbareruimte ?prefLabel
        WHERE {
          ?woonplaats a sor:Woonplaats ;
                      skos:prefLabel "Dordrecht"@nl .
          ?openbareruimte a sor:OpenbareRuimte ;
                          sor:ligtIn ?woonplaats ;
                          skos:prefLabel ?prefLabel .
          {
            SELECT ?openbareruimte (COUNT(DISTINCT ?vbo) AS ?count)
            WHERE {
              ?vbo a sor:Verblijfsobject ;
                   sor:hoofdadres/sor:ligtAan ?openbareruimte .
            }
            GROUP BY ?openbareruimte
            HAVING (COUNT(DISTINCT ?vbo) > 200)
          }
        }



    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Dordrecht',
        'novel_functions': ['MULTIPLE VARIABLES', 'GROUP BY', 'HAVING', 'SUBQUERY'],
        'new_schema_item': False,
        'number_of_relations': 5
    },

    {
        'question': 'Wat is het gemiddelde bouwjaar van gebouwen in Maastricht?',
        'sparql': """
        SELECT (AVG(YEAR(?bouwjaar)) as ?gemBouwjaar) WHERE {
            ?vbo a sor:Verblijfsobject ;
                sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats ;
                sor:maaktDeelUitVan ?gebouw .
            ?gebouw a sor:Gebouw ;
                sor:oorspronkelijkBouwjaar ?bouwjaar .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Maastricht"@nl .
        }



        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Maastricht',
        'novel_functions': ['AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 6
    },

    {
        'question': 'Wat is het gemiddelde aantal sor gemeentes dat zich bevinden in een sor provincie?',
        'sparql': """
        SELECT (AVG(?count) as ?average) WHERE {
          SELECT (COUNT(DISTINCT ?gemeente) as ?count) WHERE {
            ?gemeente a sor:Gemeente ;
              geo:sfWithin ?provincie .
            ?provincie a sor:Provincie .
          } GROUP BY ?provincie
        }



        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['AVERAGE', 'GROUP BY', 'SUBQUERY'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij de sor provincie met de meeste gemeentes. Geef mij alleen de provincie.',
        'sparql': """
        SELECT ?provincie
        WHERE {    
          ?provincie ^geo:sfWithin ?gemeente;
                     a sor:Provincie.
          ?gemeente a sor:Gemeente.  
        }
        GROUP BY ?provincie
        ORDER BY DESC(count(?gemeente))
        LIMIT 1




    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Wat is het gemiddelde aantal verblijfsobjecten dat zich in een gebouw bevinden in zwolle?',
        'sparql': """
        SELECT (AVG(?count) as ?average) WHERE {
          SELECT (COUNT(DISTINCT ?vbo) as ?count) WHERE {
            ?vbo a sor:Verblijfsobject ;
                 sor:maaktDeelUitVan ?geb .
            ?geb a sor:Gebouw .
            ?vbo sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                        skos:prefLabel "Zwolle"@nl .
          } GROUP BY ?geb
        }




        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Zwolle',
        'novel_functions': ['GROUP BY', 'SUBQUERY', 'AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 5
    },

    {
        'question': 'Geef mij de wbk gemeente met het minste aantal sor gebouwen en geef mij ook het aantal gebouwen dat hier bij hoort.',
        'sparql': """
        SELECT ?gemeente (COUNT(DISTINCT ?gebouw) AS ?aantalGebouwen)
        WHERE {
            ?gebouw a sor:Gebouw ;
                geo:sfWithin/geo:sfWithin/geo:sfWithin ?gemeente .
            ?gemeente a wbk:Gemeente.
        }
        GROUP BY ?gemeente
        ORDER BY ?aantalGebouwen
        LIMIT 1




""",
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'MULTIPLE VARIABLES'],
        'new_schema_item': False,
        'number_of_relations': 3
    },

    {
        'question': 'Geef mij de provincie waarvan de gebouwen die daarin liggen gemiddeld het oudste bouwjaar hebben. Geef mij de provincie en het gemiddelde bouwjaar van gebouwen daarin.',
        'sparql': """
SELECT ?provincie (AVG(YEAR(?bo)) AS ?avg_bouwjaar)
WHERE {
  ?building a sor:Gebouw;
            sor:oorspronkelijkBouwjaar ?bo;
            geo:sfWithin/geo:sfWithin/geo:sfWithin/^owl:sameAs/geo:sfWithin ?provincie.
}
GROUP BY ?provincie
ORDER BY ASC(?avg_bouwjaar)
LIMIT 1






""",
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'MULTIPLE VARIABLES', 'AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 6
    },

    {
        'question': 'Wat is de gemiddelde oppervlakte van percelen in Dieren?',
        'sparql': """
        SELECT (AVG(?oppervlakte) as ?gemiddeldeOppervlakte) WHERE {
            ?perceel a sor:Perceel ;
                sor:hoortBij/sor:ligtAan/sor:ligtIn ?woonplaats ;
                sor:oppervlakte ?oppervlakte .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Dieren"@nl .
        }
        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Dieren',
        'novel_functions': ['AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 5
    },

    {
        'question': 'Welke sor woonplaats bevat de sor verblijfsobjecten met gemiddeld de grootste oppervlakte? Selecteer de woonplaats en de gemiddelde oppervlakte',
        'sparql': """
        SELECT ?woonplaats (AVG(?oppervlakte) as ?gemiddeldeOppervlakte) WHERE {
            ?vbo a sor:Verblijfsobject ;
                sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats ;
                sor:oppervlakte ?oppervlakte .
            ?woonplaats a sor:Woonplaats .
        } 
        GROUP BY ?woonplaats
        ORDER BY DESC(?gemiddeldeOppervlakte)
        LIMIT 1
    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['AVERAGE', 'GROUP BY'],
        'new_schema_item': False,
        'number_of_relations': 5
    },

    {
        'question': 'Wat is het gemiddelde bouwjaar van gebouwen?',
        'sparql': """
        SELECT (AVG(YEAR(?bo)) as ?gemiddeldBouwjaarGebouwen)
        WHERE {
          ?geb a sor:Gebouw;
               sor:oorspronkelijkBouwjaar ?bo.
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['AVERAGE'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Hoeveel gebouwen zijn er waarvoor een bouwvergunning is verleend?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?gebouw) AS ?aantal) 
        WHERE {
            ?gebouw a sor:Gebouw ;
                sor:status kad-con:bouwvergunningVerleend .
        }

    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 1
    },

    {
        'question': 'Hoeveel gebouwen zijn er in gemeente Groningen waarvoor een bouwvergunning is verleend?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?gebouw) AS ?aantalGebouwen)
        WHERE {
            ?gebouw a sor:Gebouw ;
                sor:status kad-con:bouwvergunningVerleend ;
                geo:sfWithin/geo:sfWithin/geo:sfWithin ?gemeente .
            ?gemeente a wbk:Gemeente ;
                sdo0:identifier "GM0014" .
        }
""",
        'generalization_level': 'zeroshot',
        'granularity': 'Gemeente',
        'location': '0014',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 5
    },

    {
        'question': 'Hoeveel gebouwen zijn er in Tilburg waarvoor een bouwvergunning is verleend?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?gebouw) AS ?aantal)
        WHERE {
            ?gebouw a sor:Gebouw ;
                    sor:status kad-con:bouwvergunningVerleend ;
                    ^sor:maaktDeelUitVan/sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                        skos:prefLabel "Tilburg"@nl .
        }
        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Tilburg',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 5
    },

    {
        'question': 'Hoeveel sor nummeraanduidingen hebben een huisletter in Breda?',
        'sparql': """
        SELECT (COUNT(DISTINCT ?nummeraanduiding) AS ?count)
        WHERE {
            ?nummeraanduiding a sor:Nummeraanduiding ;
                sor:huisletter ?huisletter ;
                sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Breda"@nl .
        }
    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Breda',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 4
    },

    {
        'question': 'Geef mij alle sor openbare ruimtes met preflabel beginnend met een a, geef mij de openbare ruimtes en hun preflabel.',
        'sparql': """
        SELECT ?openbareRuimte ?prefLabel
        WHERE {
            ?openbareRuimte a sor:OpenbareRuimte ;
                skos:prefLabel ?prefLabel .
            FILTER (STRSTARTS(LCASE(?prefLabel), "a"))
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Geef mij alle sor openbare ruimtes met preflabel met het woord koningin erin. Geef me ook de preflabel',
        'sparql': """
        SELECT DISTINCT ?openbareRuimte ?prefLabel
        WHERE {
            ?openbareRuimte a sor:OpenbareRuimte ;
                skos:prefLabel ?prefLabel .
            FILTER(CONTAINS(LCASE(STR(?prefLabel)), "koningin"))
        }


    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING'],
        'new_schema_item': False,
        'number_of_relations': 1
    },

    {
        'question': 'Wat is de laagste en hoogste kad:bronnauwkeurigheid van sor:registratiegegevens?',
        'sparql': """
        SELECT (MIN(?nauwkeurigheid) AS ?laagsteNauwkeurigheid) (MAX(?nauwkeurigheid) AS ?hoogsteNauwkeurigheid)
        WHERE {
            ?reg a sor:Registratiegegevens ;
                 kad:bronnauwkeurigheid ?nauwkeurigheid .
        }



""",
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': [],
        'new_schema_item': True,
        'number_of_relations': 1
    },

    {
        'question': 'Wat is de meest voorkomende kad:bronbeschrijving onder de sor:registratiegegevens die kad:bronnauwkeurigheid gelijk hebben aan 0.1? Geef mij alleen de bronbeschrijving.',
        'sparql': """
        SELECT ?bronbeschrijving WHERE {
          ?reg a sor:Registratiegegevens ;
               kad:bronbeschrijving ?bronbeschrijving ;
               kad:bronnauwkeurigheid ?nauwkeurigheid .
          FILTER (?nauwkeurigheid = 0.1)
        }
        GROUP BY ?bronbeschrijving
        ORDER BY DESC(COUNT(DISTINCT ?reg))
        LIMIT 1




""",
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY'],
        'new_schema_item': True,
        'number_of_relations': 2
    },

    {
        'question': 'Wat is de meest voorkomende kad:bronbeschrijving onder de sor:registratiegegevens die kad:bronnauwkeurigheid gelijk hebben aan 20? Geef mij de bronbeschrijving en hoe vaak deze voorkomt.',
        'sparql': """
        SELECT ?bronbeschrijving (COUNT(DISTINCT ?registratie) AS ?count) WHERE {
            ?registratie a sor:Registratiegegevens ;
                kad:bronbeschrijving ?bronbeschrijving ;
                kad:bronnauwkeurigheid 20 .
        } GROUP BY ?bronbeschrijving ORDER BY DESC(?count) LIMIT 1
        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY'],
        'new_schema_item': True,
        'number_of_relations': 2
    },

    {
        'question': 'Welke sor openbare ruimte bevatten meer dan 600 kad standplaatsen? Geef mij de preflabel van de openbare ruimte, het aantal standplaatsen, en de sor woonplaats waar de openbare ruimte is gelegen.',
        'sparql': """
        SELECT ?openbareRuimteLabel (COUNT(DISTINCT ?standplaats) AS ?aantalStandplaatsen) ?woonplaatsLabel
        WHERE {
            ?standplaats a kad:Standplaats ;
                sor:hoofdadres/sor:ligtAan ?openbareRuimte .
            ?openbareRuimte a sor:OpenbareRuimte ;
                skos:prefLabel ?openbareRuimteLabel ;
                sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel ?woonplaatsLabel .
        }
        GROUP BY ?openbareRuimteLabel ?woonplaatsLabel
        HAVING (COUNT(DISTINCT ?standplaats) > 600)
        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'HAVING', 'MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 5
    },

    {
        'question': 'Welke sor openbare ruimtes, gelegen in Amsterdam, bevatten meer dan 100 kad ligtplaatsen? Geef mij de preflabel van de openbare ruimte, het aantal ligplaatsen.',
        'sparql': """
        SELECT ?openbareRuimteLabel (COUNT(DISTINCT ?ligplaats) AS ?aantalLigplaatsen)
        WHERE {
            ?woonplaats a sor:Woonplaats ;
                        skos:prefLabel "Amsterdam"@nl ;
                        ^sor:ligtIn ?openbareRuimte .
            ?openbareRuimte a sor:OpenbareRuimte ;
                            skos:prefLabel ?openbareRuimteLabel ;
                            ^sor:ligtAan/^sor:hoofdadres|^sor:nevenadres ?ligplaats .
            ?ligplaats a kad:Ligplaats .
        }
        GROUP BY ?openbareRuimteLabel
        HAVING (COUNT(DISTINCT ?ligplaats) > 100)

    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Amsterdam',
        'novel_functions': ['GROUP BY', 'HAVING', 'MULTIPLE VARIABLES'],
        'new_schema_item': True,
        'number_of_relations': 6
    },

    {
        'question': 'Wat is het gemiddelde aantal kad ligplaatsen dat zich bevinden in een sor woonplaats?',
        'sparql': """
        SELECT (AVG(?count) as ?average) WHERE {
          SELECT (COUNT(DISTINCT ?ligplaats) as ?count) ?woonplaats WHERE {
            ?ligplaats a kad:Ligplaats ;
              sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
            ?woonplaats a sor:Woonplaats .
          }
          GROUP BY ?woonplaats
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'SUBQUERY'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Wat is het gemiddelde aantal kad ligplaatsen dat zich bevinden in een sor woonplaats?',
        'sparql': """
    SELECT (AVG(?count) as ?average) WHERE {
      SELECT (COUNT(DISTINCT ?ligplaats) as ?count) ?woonplaats WHERE {
        ?ligplaats a kad:Ligplaats ;
          sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
        ?woonplaats a sor:Woonplaats .
      }
      GROUP BY ?woonplaats
    }

    """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['GROUP BY', 'SUBQUERY'],
        'new_schema_item': True,
        'number_of_relations': 3
    },

    {
        'question': 'Wat is het gemiddelde aantal sor verblijfsobjecten dat zich in een gebouw bevinden in provincie Flevoland?',
        'sparql': """
        SELECT (AVG(?count) as ?average) WHERE {
          SELECT (COUNT(DISTINCT ?vbo) as ?count) WHERE {
            ?vbo a sor:Verblijfsobject ;
                 sor:maaktDeelUitVan ?geb .
            ?geb a sor:Gebouw ;
                 geo:sfWithin/geo:sfWithin/geo:sfWithin/^owl:sameAs ?gemeente .
            ?gemeente a sor:Gemeente ;
                      geo:sfWithin provincie:0024 .
          } GROUP BY ?geb
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Provincie',
        'location': 'Flevoland',
        'novel_functions': ['GROUP BY', 'SUBQUERY'],
        'new_schema_item': False,
        'number_of_relations': 6
    },

    {
        'question': 'Geef mij de openbare ruimtes en de bijbehorende preflabels van openbare ruimtes die zich in deventer bevinden en het woord kamp in de preflabel hebben staan',
        'sparql': """
        SELECT DISTINCT ?openbareruimte ?prefLabel WHERE {
            ?openbareruimte a sor:OpenbareRuimte ;
                sor:ligtIn ?woonplaats ;
                skos:prefLabel ?prefLabel .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel "Deventer"@nl .
            FILTER(CONTAINS(LCASE(STR(?prefLabel)), "kamp"))
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Woonplaats',
        'location': 'Deventer',
        'novel_functions': ['STRING', 'MULTIPLE VARIABLES'],
        'new_schema_item': False,
        'number_of_relations': 3
    },

    {
        'question': 'Geef mij de preflabels van woonplaatsen wiens preflabels beginnen met een B. De woonplaatsen moeten minimaal 2000 verblijfsobjecten bevatten.',
        'sparql': """
        SELECT ?woonplaatsLabel WHERE {
          ?woonplaats a sor:Woonplaats ;
                      skos:prefLabel ?woonplaatsLabel .
          ?vbo a sor:Verblijfsobject ;
               sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
          FILTER (STRSTARTS(?woonplaatsLabel, "B"))
        } 
        GROUP BY ?woonplaatsLabel
        HAVING (COUNT(DISTINCT ?vbo) >= 2000)

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING', 'GROUP BY', 'HAVING'],
        'new_schema_item': False,
        'number_of_relations': 4
    },

    {
        'question': 'Geef mij de preflabels van openbare ruimtes die in woonplaatsen zijn gelegen, geef ook de preflabels van de woonplaats. Selecteer alleen openbare ruimtes en woonplaatsen die apel in hun preflabel hebben staan, beide moeten dit hebben in hun preflabel.',
        'sparql': """
        SELECT DISTINCT ?openbareRuimteLabel ?woonplaatsLabel WHERE {
            ?openbareRuimte a sor:OpenbareRuimte ;
                sor:ligtIn ?woonplaats ;
                skos:prefLabel ?openbareRuimteLabel .
            ?woonplaats a sor:Woonplaats ;
                skos:prefLabel ?woonplaatsLabel .
            FILTER (CONTAINS(LCASE(?openbareRuimteLabel), "apel"))
            FILTER (CONTAINS(LCASE(?woonplaatsLabel), "apel"))
        }

        """,
        'generalization_level': 'zeroshot',
        'granularity': 'Nederland',
        'location': 'Nederland',
        'novel_functions': ['STRING'],
        'new_schema_item': False,
        'number_of_relations': 3
    },

]
file_path = "development_set.json"

# Write the development_set list to a JSON file
with open(file_path, "w") as json_file:
    json.dump(development_set, json_file, indent=4)

file_path = "test_set.json"

# Write the development_set list to a JSON file
with open(file_path, "w") as json_file:
    json.dump(test_set, json_file, indent=4)

nederland_counter = 0
provincie_counter = 0
gemeente_counter = 0
woonplaats_counter = 0
for item in test_set:
    if item['granularity'] == 'Nederland':
        nederland_counter = nederland_counter +1
    if item['granularity'] == 'Woonplaats':
        woonplaats_counter = woonplaats_counter +1
    if item['granularity'] == 'Gemeente':
        gemeente_counter = gemeente_counter +1
    if item['granularity'] == 'Provincie':
        provincie_counter = provincie_counter +1



print(nederland_counter)
print(woonplaats_counter)
print(gemeente_counter)
print(provincie_counter)

