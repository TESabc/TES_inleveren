import random
import string
import requests

from create_training_data.query import Query
from create_training_data.functions import Functions
from create_training_data.dependencies import Dependency, get_dependency_text
from create_training_data.query_components import (SpecialTokens, get_template_formatted, Template, STREET, CITY,
                                                     WIJK, BUURT,
                                                     GEMEENTE, NEWEST, OLDEST, BIGGEST, SMALLEST, PARCEL_SURFACE,
                                                     AMOUNT, BUILD_YEAR,
                                                     HOUSE_SURFACE, brt_properties, EQUAL, PROVINCIE, LAND, HOUSES,
                                                     SMALLER, BIGGER,
                                                     PARCEL, BEFORE, AFTER, SQUARE_METER, ACCOMODATION_OBJECTS, SURFACE,
                                                     MONUMENTAL_STATUS, CHANGE_ADDRESS_SUGGESTION, PREMISES, brt_properties_to_kad_con)

"""
This file provides functionality for generating training data that pairs natural language questions 
with their corresponding SPARQL queries. 
"""

class BrtFilterQuery(Query):
    _weight = 3
    _dependencies = {Dependency.ADDRESS}

    year_filter_synonyms = {
        'groter': AFTER,
        'gelijk': EQUAL + [None],
        'kleiner': BEFORE,
    }

    size_filter_synonyms = {
        'groter': BIGGER,
        'gelijk': EQUAL + [None],
        'kleiner': SMALLER,
    }

    filter_values = {
        'groter': '>',
        'gelijk': '=',
        'kleiner': '<',
    }

    order_synonyms = {
        'newest': NEWEST,
        'oldest': OLDEST,
        'biggest': BIGGEST,
        'smallest': SMALLEST,
    }

    order_values = {
        'newest': 'DESC',
        'oldest': 'ASC',
        'biggest': 'DESC',
        'smallest': 'ASC',
    }

    granularity_synonyms = {
        'straat': STREET,
        'buurt': BUURT,
        'wijk': WIJK,
        'woonplaats': CITY,
        'gemeente': GEMEENTE,
        'provincie': PROVINCIE,
        'land': LAND + [None],
    }

    property_synonyms = {
        'huizen': HOUSES,
        'percelen': PARCEL,
        'gebouwen': PREMISES,
        'monumenten': MONUMENTAL_STATUS,
        'verblijfsobjecten': ACCOMODATION_OBJECTS,
    }

    '''
    The constructor of this class calls the constructor of the superclass.

    We put in template = 'brt_filter_query' to load the contents of the appropriate YAML file into
    self.questions.
    '''

    def __init__(self, template='brt_filter_query', *args, **kwargs):
        super().__init__(template=template, *args, **kwargs)

    '''
    We basically input an answer_kwargs into this function.

    1) if adress_data is in the keys of the answer_kwargs:
    We check whether own_adress is in answer kwargs:
    1a) If own_adress is in answer_kwargs (and it returns a truthy value)
    We randomly select one of the granularity keys
        'straat': STREET,
        'buurt': BUURT,
        'wijk': WIJK,
        'woonplaats': CITY,
        'gemeente': GEMEENTE,
        'provincie': PROVINCIE,
        'land': LAND + [None]
    which is not equal to the granularity_key in our answer_kwargs

    We return a tuple of which the first element tells us that we have a granularity_key and the second
    element is the randomly selected granularity_key (which is not equal to what was in answer_kwargs)
    1b) If own_adress is not in answer_kwargs (or it returns a falsy value)
    -there a 25% chance that we return a tuple with the following two strings:
    ('amount_key', 'aantal')
    -there is a 75% chance that we return a tuple of which the first element tells us we have a 
    property_key and the second element is a randomly selected property_key which is not equal to the
    property key that was already in the answer kwargs.

    2) if adress_data is not one of the keys in answer_kwargs
    the function simply returns None    
    '''

    def get_related(self, answer_kwargs):
        if 'address_data' in answer_kwargs:
            if answer_kwargs.get('own_address'):
                key = random.choice([
                    key for key in self.granularity_synonyms.keys() if key != answer_kwargs.get('granularity_key')
                ])
                result = ('granularity_key', key)
            else:
                if random.randint(0, 100) > 75:
                    result = ('amount_key', 'aantal')
                else:
                    key = random.choice([
                        key for key in self.property_synonyms.keys() if key != answer_kwargs.get('property_key')
                    ])
                    result = ('property_key', key)
        else:
            result = None

        return result

    def get_related_options(self, answer_kwargs, query_variant_stack=None, *args, **kwargs):
        if 'address_data' in answer_kwargs:
            if answer_kwargs.get('own_address'):
                keys_list = list(self.granularity_synonyms.keys())
                try:
                    index = keys_list.index(answer_kwargs.get('granularity_key')) + 1
                except ValueError:
                    index = 0

                variants = keys_list[index: index + 3]
                if len(variants) < 3:
                    variants += keys_list[:3 - len(variants)]
                variants.append(CHANGE_ADDRESS_SUGGESTION)
            else:
                keys_list = list(self.property_synonyms.keys())
                try:
                    index = keys_list.index(answer_kwargs.get('property_key')) + 1
                except ValueError:
                    index = 0

                variants = keys_list[index: index + 3]
                if len(variants) < 3:
                    variants += keys_list[:3 - len(variants)]
                variants.append('aantal')

            result = f' {SpecialTokens.SEPARATOR.value} '.join(variants)
        else:
            result = ''

        return result

    '''
    This is the function that general a SPARQL query.


    '''

    def get_answer(self, address_data, property_key, granularity_key, order_key, brt_property_key,
                   year_filter_options=None, year_filter_values=None, house_filter_options=None,
                   house_filter_values=None, parcel_filter_options=None, parcel_filter_values=None, amount_key=None,
                   extra_return_key=None,
                   *args, **kwargs):

        # we create sets for the golden classes and golden relations, so that we can let the function
        # return the golden classes and relations that are used in SPARQL query that is generated.
        golden_classes = set()
        golden_relations = set()
        # We create a string sentence of the adress
        address = ' '.join(address_data.values()) if isinstance(address_data, dict) else address_data

        '''
        Here we see whether we need the oldest/newest biggest/smallest thing retrieved.
        This all has effects on how the query needs to be written.

        When our order_key is oldest or newest we always do this on ?bo (oorspronkelijk bouwjaar)
        When we are talking about biggest or smallest we will be working with perceeloppervlakte or woningoppervlakte
        ( depending on the property_key )



        '''
        if order_key in ('oldest', 'newest'):
            key = '?bo'
        elif order_key in ('biggest', 'smallest'):
            key = '?po' if property_key == 'percelen' else '?wo'
        else:
            key = None

        if key:
            if order_key:
                order = f'ORDER BY {self.order_values[order_key]} ({key}) LIMIT 1'
            else:
                order = 'LIMIT 9999'
        else:
            order = 'LIMIT 9999'

        '''
        Here we add filters on oorspronkelijkBouwjaar, woningoppervlakte, perceeloppervlakte if needed.

        Note that when we are working with an interval we add two seperate filters.

        The different filter options and values automatically determine on which variable we will
        need to filter.

        Note that we start to create a list with filter strings here.
        '''
        filters = []
        for (year_filter_option, year_filter_value) in zip(year_filter_options, year_filter_values):
            filters.append(f"FILTER (str(?bo) {self.filter_values[year_filter_option]} '{year_filter_value}')")
        for (house_filter_option, house_filter_value) in zip(house_filter_options, house_filter_values):
            filters.append(f"FILTER (?wo {self.filter_values[house_filter_option]} {house_filter_value})")
        for (parcel_filter_option, parcel_filter_value) in zip(parcel_filter_options, parcel_filter_values):
            filters.append(f"FILTER (?po {self.filter_values[parcel_filter_option]} {parcel_filter_value})")

        '''
        If the brt_property_key we are working with is voorziening, we EXCLUDE huizenblok through an additional filter
        that gets added to our list.

        We also put a building_part of the query which will allow us to retrieve the variable that we will filter on
        '''

        if brt_property_key is not None:

            building_part = """
                ?gebz sor:hoortBij ?vbo; 
                    kad:gebouwtype %s.
            """ % (brt_properties_to_kad_con[brt_property_key])
            golden_relations.add('sor:hoortBij')
            golden_relations.add('kad:gebouwtype')

        else:
            building_part = ''

        # We change the filters such that it becomes one big string
        filters = '\n'.join(filters)

        '''
        IN THE FIRST PART OF THE IF-STATEMENT WE WILL BE DEALING WITH CASES WHERE THE GRANULARITY IS
        straat or woonplaats
        '''

        if granularity_key in ('straat', 'woonplaats'):

            # First we link our data to a certain street IF our granularity is street
            if granularity_key == 'straat':
                street = "skos:prefLabel \"#straatnaam\"@nl;"
                golden_relations.add('skos:prefLabel')
            else:
                street = ''

            # changed this part to what we have above
            # street = "skos:prefLabel \"#straatnaam\"@nl;" if granularity_key == 'straat' else ''

            # I suspect these are the variables that our query eventually will be returning
            return_keys = []

            # If we are working with percelen we retrieve that data in subject here.
            if property_key == 'percelen':

                # Note that we dynamically determine whether certain parts of the sparql query are strictly needed:

                # If we have one of these options we will need to retrieve vbo's and link them to percelen through
                # ?na.
                if brt_property_key or house_filter_options or year_filter_options:
                    vbo_part = "?vbo a sor:Verblijfsobject;"
                    na_part = "sor:hoofdadres ?na;"
                    golden_relations.add('sor:hoofdadres')
                    golden_classes.add('sor:Verblijfsobject')

                    # If we have house_filter_options we will need the variable ?wo
                    if house_filter_options:
                        wo_part = "sor:oppervlakte ?wo;"
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # If we have year_filter options we will need to link the vbo's to ?geb, and also link ?geb to ?bo.
                    if year_filter_options:
                        vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                        bo_part = """
                        ?geb
                            sor:oorspronkelijkBouwjaar ?bo.
                        """
                        golden_relations.add('sor:maaktDeelUitVan')
                        golden_relations.add('sor:oorspronkelijkBouwjaar')
                    else:
                        vbo_geb_koppeling_part = ''
                        bo_part = ''

                    # Now we make sure we have the "." instead of the ";" in the right location in the sparql query

                    # In this case we only need vbo's that are linked to our parcel in order to be able to put a
                    # brt_filter. (note we left out the case where all are not necessary and all are necessary,
                    # those are already taken care of before)
                    if vbo_geb_koppeling_part == '' and wo_part == '' and brt_property_key:
                        na_part = 'sor:hoofdadres ?na.'
                    # We check all other cases and take action if necessary
                    elif vbo_geb_koppeling_part == '' and wo_part != '' and not brt_property_key:
                        wo_part = "sor:oppervlakte ?wo."
                    elif vbo_geb_koppeling_part != '' and wo_part == '' and not brt_property_key:
                        vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                    elif vbo_geb_koppeling_part != '' and wo_part != '' and not brt_property_key:
                        pass
                    elif vbo_geb_koppeling_part != '' and wo_part == '' and brt_property_key:
                        pass
                    elif vbo_geb_koppeling_part == '' and wo_part != '' and brt_property_key:
                        wo_part = "sor:oppervlakte ?wo."

                else:
                    vbo_part = ''
                    na_part = ''
                    wo_part = ''
                    vbo_geb_koppeling_part = ''
                    bo_part = ''

                # We dynamically include or exclude parts from the last block (about percelen)
                if key == '?po' or parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    po_part = 'sor:oppervlakte ?po;'
                    golden_relations.add('sor:oppervlakte')
                else:
                    po_part = ''

                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # We again make sure we have the "." in the correct location for the last block now

                # If no geo_part and no po_part we simply put the dot after ?na, since we always need that
                # to be able to link our percelen to a location.
                if geo_part == '' and po_part == '':
                    per_na_part = 'sor:hoortBij ?na.'
                # If one of them is not empty we will put a semicolon indicating more is to come.
                else:
                    per_na_part = 'sor:hoortBij ?na;'

                # if geo_part is empty but po_part is not than we put the dot after ?po
                if geo_part == '' and po_part != '':
                    po_part = 'sor:oppervlakte ?po.'
                # Note that if geo_part is NOT empty and po_part is empty the dot is already in the correct place.

                '''
                                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:oppervlakte ?wo;
                    sor:maaktDeelUitVan ?geb.

                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.
                """
                '''

                golden_classes.add('sor:Perceel')
                subject = """
                %s
                    %s
                    %s
                    %s

                %s


                ?per 
                    a sor:Perceel;
                    %s
                    %s
                    %s
                """ % (vbo_part, na_part, wo_part, vbo_geb_koppeling_part, bo_part, per_na_part, po_part, geo_part)

                # If we are not counting anything, we want to return all these 3 variables it seems
                if amount_key is None:
                    return_keys.extend(['?per'])

                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])

                # If we are counting something we will count the amount of percelen
                else:
                    return_keys.append('(COUNT(DISTINCT ?per) as ?aantal)')
            elif property_key == 'monumenten':

                # Note that we dynamically determine whether certain parts of the sparql query are strictly needed:

                # if one of these things is the case we need the variable ?wo
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                    wo_part = "sor:oppervlakte ?wo;"
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # If we do not have the variable ?wo we simply end the first block with a dot
                # after the retrieval of ?geb because we then already have linked our ?geb to a ?na.
                if wo_part == '':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')
                    wo_part = "sor:oppervlakte ?wo."

                # if we need ?bo we set it equal to this.
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = "sor:oorspronkelijkBouwjaar ?bo;"
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # if we need geometry we set it equal to this.
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # If we need neither ?bo and geometry we set the dot after grondslag
                if geo_part == '' and bo_part == '':
                    grondslag_part = '^kad:gevestigdOp/kad:grondslag ?grondslag.'
                    golden_relations.add('kad:gevestigdOp')
                    golden_relations.add('kad:grondslag')

                else:
                    grondslag_part = '^kad:gevestigdOp/kad:grondslag ?grondslag;'
                    golden_relations.add('kad:gevestigdOp')
                    golden_relations.add('kad:grondslag')

                # If we do not need geometry but we do need ?bo we put the dot after ?bo instead
                if geo_part == '' and bo_part != '':
                    bo_part = "sor:oorspronkelijkBouwjaar ?bo."

                # note if we need both (geometry & ?bo) everything is already good

                # note if we need geometry but not ?bo everything is also already good.

                # If we need ?po we add this additional block dynamically
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                        OPTIONAL { 
                        ?per a sor:Perceel;
                            sor:hoortBij ?na;
                            sor:oppervlakte ?po.
                        }
                        """
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                    golden_classes.add('sor:Perceel')

                else:
                    per_part = ''

                '''
                subject = """
                VALUES ?grondslag {
                    kad-con:GG 
                    kad-con:GWA 
                }

                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo.


                ?geb a sor:Gebouw
                    ^kad:gevestigdOp/kad:grondslag ?grondslag;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_relations.add('sor:hoofdadres')
                golden_classes.add('sor:Verblijfsobject')
                subject = """
                    VALUES ?grondslag {
                        kad-con:GG 
                        kad-con:GWA 
                    }

                    ?vbo a sor:Verblijfsobject;
                        sor:hoofdadres ?na;
                        %s
                        %s


                    ?geb a sor:Gebouw
                        %s
                        %s
                        %s

                    %s 

                    """ % (vbo_geb_koppeling_part, wo_part, grondslag_part, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            # When we are working with huizen as our property_key
            # Then the verblijfsobject clearly should have a woonfunctie as gebruiksdoel
            elif property_key == 'huizen':

                # If we have any of these we will need ?wo
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                    wo_part = "sor:oppervlakte ?wo;"
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # If we have any of these we will need ?geb linked to our vbo's and also need ?bo linked to ?geb
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb;'
                    geb_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:maaktDeelUitVan')
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    vbo_geb_koppeling_part = ''
                    geb_part = ''

                # If we have this we need the geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                na_part = 'sor:hoofdadres ?na;'
                # We put the dot in the correct place according to all the different cases.
                # note that there are (3 choose 1) + (3 choose 2) + 2*(3 choose 3) = 8 options in total
                # If all of these are empty we put the dot directly after ?na
                if wo_part == '' and vbo_geb_koppeling_part == '' and geo_part == '':
                    na_part = 'sor:hoofdadres ?na.'
                elif wo_part != '' and vbo_geb_koppeling_part != '' and geo_part == '':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                elif wo_part != '' and vbo_geb_koppeling_part == '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part != '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part == '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part != '' and geo_part == '':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                elif wo_part != '' and vbo_geb_koppeling_part == '' and geo_part == '':
                    wo_part = "sor:oppervlakte ?wo."
                elif wo_part != '' and vbo_geb_koppeling_part != '' and geo_part != '':
                    pass

                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                    golden_classes.add('sor:Perceel')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:gebruiksdoel sor-con:woonfunctie;
                    sor:hoofdadres ?na;
                    sor:oppervlakte ?wo;
                    sor:maaktDeelUitVan ?geb;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.



                ?geb a sor:Gebouw
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:gebruiksdoel')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:gebruiksdoel sor-con:woonfunctie;
                    %s
                    %s
                    %s
                    %s



                %s

                %s
                """ % (na_part, wo_part, vbo_geb_koppeling_part, geo_part, geb_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')
            elif property_key == 'verblijfsobjecten':

                # If we have any of these we will need ?wo
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                    wo_part = "sor:oppervlakte ?wo;"
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # If we have any of these we will need ?geb linked to our vbo's and also need ?bo linked to ?geb
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb;'
                    geb_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:maaktDeelUitVan')
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    vbo_geb_koppeling_part = ''
                    geb_part = ''

                # If we have this we need the geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                na_part = 'sor:hoofdadres ?na;'
                # We put the dot in the correct place according to all the different cases.
                # note that there are (3 choose 1) + (3 choose 2) + 2*(3 choose 3) = 8 options in total
                # If all of these are empty we put the dot directly after ?na
                if wo_part == '' and vbo_geb_koppeling_part == '' and geo_part == '':
                    na_part = 'sor:hoofdadres ?na.'
                elif wo_part != '' and vbo_geb_koppeling_part != '' and geo_part == '':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                elif wo_part != '' and vbo_geb_koppeling_part == '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part != '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part == '' and geo_part != '':
                    pass
                elif wo_part == '' and vbo_geb_koppeling_part != '' and geo_part == '':
                    vbo_geb_koppeling_part = 'sor:maaktDeelUitVan ?geb.'
                elif wo_part != '' and vbo_geb_koppeling_part == '' and geo_part == '':
                    wo_part = "sor:oppervlakte ?wo."
                elif wo_part != '' and vbo_geb_koppeling_part != '' and geo_part != '':
                    pass

                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                    golden_classes.add('sor:Perceel')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:oppervlakte ?wo;
                    sor:maaktDeelUitVan ?geb;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.

                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    %s
                    %s
                    %s
                    %s

                %s

                %s
                """ % (na_part, wo_part, vbo_geb_koppeling_part, geo_part, geb_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])

                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')
            else:  # 'gebouwen'

                # here we determine whether we need ?wo variable.
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                    wo_part = "sor:oppervlakte ?wo."
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # Note that we always need everything up untill ?geb. This because we need ?na to link it to a location
                # within this if-statement and .
                # So here we put the dot directly after ?geb if we do not need ?wo
                if wo_part == '':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb.'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')

                # we again determine if we need ?bo
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo;'
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # determine if we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # Here we determine where we should place the dot in the query block about ?geb
                geb = '?geb a sor:Gebouw;'
                golden_classes.add('sor:Gebouw')
                if geo_part == '' and bo_part != '':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo.'
                elif geo_part != '' and bo_part == '':
                    pass
                elif geo_part == '' and bo_part == '':
                    geb = '?geb a sor:Gebouw.'

                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                    golden_classes.add('sor:Perceel')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo.


                ?geb a sor:Gebouw;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:hoofdadres')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    %s
                    %s


                %s
                    %s
                    %s

                %s
                """ % (vbo_geb_koppeling, wo_part, geb, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])

                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            return_keys = ' '.join(return_keys)

            '''
            Note that whether we are dealing with street or woonplaats we always need a woonplaats in the query.

            the building_part allows us to put "filters" on brt_property_key.
            '''
            golden_classes.add('sor:Woonplaats')
            golden_classes.add('sor:OpenbareRuimte')
            golden_relations.add('skos:prefLabel')
            golden_relations.add('sor:ligtIn')
            golden_relations.add('sor:ligtAan')
            query = """
            SELECT DISTINCT %s
            WHERE {
                %s
                %s

                ?woonplaats a sor:Woonplaats;
                    skos:prefLabel "#woonplaatsnaam"@nl;
                    ^sor:ligtIn ?openbareruimte.

                ?openbareruimte a sor:OpenbareRuimte;
                    %s
                    ^sor:ligtAan ?na.

                %s
            } %s
            """ % (return_keys, subject, building_part, street, filters, order)



        # HERE WE TREAT THE OTHER GRANULARITIES
        elif granularity_key in ('wijk', 'buurt', 'gemeente', 'provincie'):
            return_keys = []

            # The subject is the same as before for other granularities
            if property_key == 'percelen':
                # Note that in this part although we are focused on percelen, we will always have to link percelen to
                # vbo's through ?na in order to then also provide a link to ?geb on which we can link it to
                # a location within these granularities through ?geb

                # We determine if we need the variable ?wo
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                    wo_part = "sor:oppervlakte ?wo."
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # We determine where we need to put the dot
                if wo_part == '':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb.'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')

                # Here we determine if we need to get ?bo (note we already obtained ?geb in this first block)
                if year_filter_options:
                    bo_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # We determine whether we need ?po
                if key == '?po' or parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    po_part = 'sor:oppervlakte ?po;'
                    golden_relations.add('sor:oppervlakte')
                else:
                    po_part = ''

                # we determine whether we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # We again make sure we have the "." in the correct location for the last block now
                if geo_part == '' and po_part == '':
                    per_na_part = 'sor:hoortBij ?na.'
                    golden_relations.add('sor:hoortBij')
                else:
                    per_na_part = 'sor:hoortBij ?na;'
                    golden_relations.add('sor:hoortBij')
                if geo_part == '' and po_part != '':
                    po_part = 'sor:oppervlakte ?po.'
                # Note if geo part is not empty and po part is empty everything is already fine
                # furthermore, if both are not empty then everything is also already fine

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo.


                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:hoofdadres')
                golden_classes.add('sor:Perceel')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    %s
                    %s

                %s

                ?per a sor:Perceel;
                    %s
                    %s
                    %s
                """ % (vbo_geb_koppeling, wo_part, bo_part, per_na_part, po_part, geo_part)

                if amount_key is None:
                    return_keys.extend(['?per'])

                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?per) as ?aantal)')


            # The subject is the same as before for other granularities
            elif property_key == 'monumenten':

                # If one of these things evaluates to True we need (part) of the ?vbo block
                # note that we also need if we have a brt_property_key
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options or parcel_filter_options or extra_return_key == 'perceeloppervlakte' or brt_property_key:
                    vbo_main_part = """
                    ?vbo a sor:Verblijfsobject;
                        sor:maaktDeelUitVan ?geb;
                    """
                    golden_classes.add('sor:Verblijfsobject')
                    golden_relations.add('sor:maaktDeelUitVan')

                    # we determine whether we need ?wo
                    if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                        wo_part = "sor:oppervlakte ?wo;"
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # we determine if we need ?na in order to provide a linkage to percelen
                    if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                        vbo_na_part = "sor:hoofdadres ?na."
                        golden_relations.add('sor:hoofdadres')
                    else:
                        vbo_na_part = ''

                    # If we do not need any of these things we put the dot at the beginning
                    # (note that in this case we only need the ?vbo variable to apply a brt_property_key filter)
                    if vbo_na_part == '' and wo_part == '':
                        vbo_main_part = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb.
                        """
                    # If one of these is not empty we will apply a semicolon because there is more to come
                    else:
                        vbo_main_part = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb;
                        """

                        # Now we know we need at least one of the following parts.
                        # Note only in the following case do we need to change the dot placement
                        if vbo_na_part == '' and wo_part != '':
                            wo_part = "sor:oppervlakte ?wo."


                else:
                    # In this part of the if-statement we do not need anything related to ?vbo so we keep everything empty
                    vbo_main_part = ''
                    wo_part = ''
                    vbo_na_part = ''

                # We determine if we need ?bo again
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo;'
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # we determine if we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # If both parts are empty we place the dot after the grondslag part which we always need
                # to only retrieve monuments
                if geo_part == '' and bo_part == '':
                    grondslag_part = '^kad:gevestigdOp/kad:grondslag ?grondslag.'
                    golden_relations.add('kad:gevestigdOp')
                    golden_relations.add('kad:grondslag')
                else:
                    grondslag_part = '^kad:gevestigdOp/kad:grondslag ?grondslag;'
                    golden_relations.add('kad:gevestigdOp')
                    golden_relations.add('kad:grondslag')

                # If we need the ?bo parts but the the geometry part we place the dot here
                if geo_part == '' and bo_part != '':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo.'

                # Note that for the case where we need geometry and not ?bo the dot placement is already correct.

                # Als for the case that both are not empty the dot placement is also already correct.

                # The only reason currently we need parcels is when we need it for ?po
                # Note you might want to change this later if you want to retrieve only property IRI's and parcel IRI's
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                VALUES ?grondslag {
                    kad-con:GG 
                    kad-con:GWA 
                }

                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na.

                ?geb a sor:Gebouw;
                    ^kad:gevestigdOp/kad:grondslag ?grondslag;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Gebouw')
                subject = """
                VALUES ?grondslag {
                    kad-con:GG 
                    kad-con:GWA 
                }

                %s
                    %s
                    %s

                ?geb a sor:Gebouw;
                    %s
                    %s
                    %s

                %s
                """ % (vbo_main_part, wo_part, vbo_na_part, grondslag_part, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])
                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            # The subject is the same as before for the other granularities
            elif property_key == 'huizen':

                # note that we always need part of ?vbo because we assume huizen = woningen and these are vbo's

                # We determine if we need ?wo
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                    wo_part = "sor:oppervlakte ?wo;"
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # We determine if we need ?po. If we do, we also retrieve ?na here in order to link our vbo's to
                # ?per to be able to retrieve ?po later.
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    vbo_na_part = "sor:hoofdadres ?na;"
                    golden_relations.add('sor:hoofdadres')
                else:
                    vbo_na_part = ''

                # We determine whether we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = "geo:hasGeometry/geo:asWKT ?geo_wgs84."
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # Here we go through all the cases to put the dots at the correct place.
                # First we put the dot after woonfunctie directly if we do not need any of the other parts
                if vbo_na_part == '' and wo_part == '' and geo_part == '':
                    woonfunctie_part = 'sor:gebruiksdoel sor-con:woonfunctie.'
                    golden_relations.add('sor:gebruiksdoel')
                else:
                    # In this part of the if-statement we know we need at least one of the parts, so we place
                    # a semicolon after woonfunctie because we know there will be more to come.
                    woonfunctie_part = 'sor:gebruiksdoel sor-con:woonfunctie;'
                    golden_relations.add('sor:gebruiksdoel')

                    # we skip the option where all 3 are non-empty because the dot placement is already correct for
                    # that case

                    if geo_part != '' and vbo_na_part == '' and wo_part != '':
                        pass  # dot placement already correct
                    elif geo_part != '' and vbo_na_part != '' and wo_part == '':
                        pass  # dot placement already correct
                    elif geo_part == '' and vbo_na_part != '' and wo_part != '':
                        vbo_na_part = "sor:hoofdadres ?na."  # change dot placement
                    elif geo_part == '' and vbo_na_part == '' and wo_part != '':
                        wo_part = "sor:oppervlakte ?wo."  # change dot placement
                    elif geo_part == '' and vbo_na_part != '' and wo_part == '':
                        vbo_na_part = "sor:hoofdadres ?na."  # change dot placement
                    # note in the last case where only geo_part is not empty we also don't need to change dot placement.
                    # we could have put another 'elif' statement with a 'pass' keyword but that is not needed

                # note we always need ?geb in order a link it to as specific place in our granularity.
                # but we already retrieved ?geb in this first block about ?vbo
                # so we determine here whether we need ?bo only
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = """
                    ?geb a sor:Gebouw
                        sor:oorspronkelijkBouwjaar ?bo.
                    
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                    golden_classes.add('sor:Gebouw')
                else:
                    bo_part = ''

                # we determine whether we will be using ?po, and add the block dynamically
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:gebruiksdoel sor-con:woonfunctie;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.


                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:maaktDeelUitVan')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    %s
                    %s
                    %s
                    %s


                %s


                %s
                """ % (woonfunctie_part, wo_part, vbo_na_part, geo_part, bo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')

            # the subject is the same as before for other granularities
            elif property_key == 'verblijfsobjecten':

                # note since our subject is vbo's we always need to get vbo's

                # Here we determine whether we need ?wo
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                    wo_part = "sor:oppervlakte ?wo;"
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # Here we determine if we need ?po. If we need ?po, we retrieve ?na to be able to make the hops later
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    vbo_na_part = "sor:hoofdadres ?na;"
                    golden_relations.add('sor:hoofdadres')
                else:
                    vbo_na_part = ''

                # we determine if we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = "geo:hasGeometry/geo:asWKT ?geo_wgs84."
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # here we make sure our dot placement is correct
                # If all 3 parts are empty we put the dot after ?geb
                if vbo_na_part == '' and wo_part == '' and geo_part == '':
                    vbo_geb_part = 'sor:maaktDeelUitVan ?geb.'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    # at least one of the 3 parts is not empty so we put a semicolon because there is more to come
                    vbo_geb_part = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')

                    # we skip the case where all parts are not-empty because in that case the dot placement is already correct

                    # Note we only consider the cases where the dot placement actually changes for efficiency this time.
                    if geo_part == '' and vbo_na_part == '' and wo_part != '':
                        wo_part = "sor:oppervlakte ?wo."
                    elif geo_part == '' and vbo_na_part != '' and wo_part == '':
                        vbo_na_part = "sor:hoofdadres ?na."
                    elif geo_part == '' and vbo_na_part != '' and wo_part != '':
                        vbo_na_part = "sor:hoofdadres ?na."

                # We determine whether we need ?bo variable here
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = """
                    ?geb a sor:Gebouw
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                    golden_classes.add('sor:Gebouw')
                else:
                    bo_part = ''

                # we also retrieve the ?per block if we need the variable ?po
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.

                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:maaktDeelUitVan')
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    %s
                    %s
                    %s

                %s

                %s
                """ % (wo_part, vbo_na_part, geo_part, bo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')

                # Same subject as before for other granularities
            else:  # 'gebouwen'

                '''
                I THINK WE CAN REMOVE THIS ENTIRE PART LATER! NO IDEA WHY IT IS HERE...

                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                    wo_part = "sor:oppervlakte ?wo;"
                else:
                    wo_part = ''

                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    vbo_na_part = "sor:hoofdadres ?na;"
                else:
                    vbo_na_part = ''

                if extra_return_key == 'geometrie':
                    geo_part = "geo:hasGeometry/geo:asWKT ?geo_wgs84."

                if vbo_na_part == '' and wo_part == '' and geo_part == '':
                    vbo_geb_part = 'sor:maaktDeelUitVan ?geb.'
                else:
                    vbo_geb_part = 'sor:maaktDeelUitVan ?geb;'


                    if geo_part == '' and vbo_na_part == '' and wo_part != '':
                        wo_part = "sor:oppervlakte ?wo."
                    elif geo_part == '' and vbo_na_part != '' and wo_part == '':
                        vbo_na_part = "sor:hoofdadres ?na."
                    elif geo_part == '' and vbo_na_part != '' and wo_part != '':
                        vbo_na_part = "sor:hoofdadres ?na."

                '''

                # If one of these evaluates to True we will need the ?vbo block
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options or parcel_filter_options or extra_return_key == 'perceeloppervlakte' or brt_property_key:

                    # we determine if we need ?wo
                    if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                        wo_part = "sor:oppervlakte ?wo;"
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # we determine if we need ?po (and therefore need ?na through which we can link our gebouwen to percelen)
                    if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                        vbo_na_part = "sor:hoofdadres ?na."
                        golden_relations.add('sor:hoofdadres')
                    else:
                        vbo_na_part = ''

                    # In this case we only need vbo's (linked to ?geb) in order to apply a brt_property_key filter
                    # so we put a dot behind ?geb
                    if vbo_na_part == '' and wo_part == '' and brt_property_key:
                        vbo_main_part = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb.
                        """
                        golden_classes.add('sor:Verblijfsobject')
                        golden_relations.add('sor:maaktDeelUitVan')
                    # Note within this block we know we need the vbo block.
                    # If vbo_na_part and wo_part are not empty we put a semicolon after ?geb because there is more to come
                    else:
                        vbo_main_part = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb;
                        """
                        golden_classes.add('sor:Verblijfsobject')
                        golden_relations.add('sor:maaktDeelUitVan')
                        # Now we determine where to put the dot.
                        # note in the case where both are not empty, the dots are already on the correct place.
                        # in the case where vbo_na_part is not empty and wo_part is empty the dots are also in the correct place
                        # this is the only case where things actually change:
                        if vbo_na_part == '' and wo_part != '':
                            wo_part = "sor:oppervlakte ?wo."


                else:
                    # in this case we need nothing of the ?vbo block
                    vbo_main_part = ''
                    wo_part = ''
                    vbo_na_part = ''

                # we determine whether we need ?bo
                if year_filter_options or key == '?bo' or extra_return_key == 'bouwjaar':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo;'
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # we determine if we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # if both are empty we put the dot directly after sor:Gebouw
                if geo_part == '' and bo_part == '':
                    geb_part = '?geb a sor:Gebouw.'
                    golden_classes.add('sor:Gebouw')
                # otherwise we put a semicolon because there is more to come
                else:
                    geb_part = '?geb a sor:Gebouw;'
                    golden_classes.add('sor:Gebouw')

                # This is the only case we care about
                # if both not empty dots are already correct
                # if only geo_part not empty then the dots are also already correct
                if geo_part == '' and bo_part != '':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo.'

                # we determine if we need anything with parcels (percelen)
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    # if we need ?po we add it
                    if parcel_filter_options:
                        per_part = """
                        OPTIONAL { 
                        ?per a sor:Perceel;
                            sor:hoortBij ?na;
                            sor:oppervlakte ?po.
                        }
                        """
                        golden_classes.add('sor:Perceel')
                        golden_relations.add('sor:hoortBij')
                        golden_relations.add('sor:oppervlakte')
                    # if we don't need ?po we don't add it
                    else:
                        per_part = """
                        OPTIONAL { 
                        ?per a sor:Perceel;
                            sor:hoortBij ?na.
                        }
                        """
                        golden_classes.add('sor:Perceel')
                        golden_relations.add('sor:hoortBij')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na.

                ?geb a sor:Gebouw;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                subject = """
                %s
                    %s
                    %s

                %s
                    %s
                    %s

                %s
                """ % (vbo_main_part, wo_part, vbo_na_part, geb_part, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])


                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            return_keys = ' '.join(return_keys)

            '''
            THIS IS WHERE THINGS GET DIFFERENT FOR THESE SPECIFIC GRANULARITIES

            -First note that we do not have a main query part.
            In the previous part we needed that to go
            from nummeraanduiding to openbareruimte to woonplaats

            -We have different granularity_keys's
            to go from building to the appropriate granularity

            note that we do multiple jump through the properties.





            '''

            if granularity_key == 'land':
                granularity = ''

            elif granularity_key == 'provincie':
                granularity = """
                ?gemeente geo:sfWithin provincie:#provinciecode.
                ?gemeente owl:sameAs/^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
                """
                golden_relations.add('geo:sfWithin')
                golden_relations.add('owl:sameAs')
            elif granularity_key == 'gemeente':
                granularity = """
                ?gemeente sdo0:identifier "#gemeentecode".
                ?gemeente ^geo:sfWithin/^geo:sfWithin/^geo:sfWithin ?geb.
                """
                golden_relations.add('sdo0:identifier')
                golden_relations.add('geo:sfWithin')
            elif granularity_key == 'buurt':
                granularity = """
                ?buurt sdo0:identifier "#buurtcode".
                ?buurt ^geo:sfWithin ?geb.
                """
                golden_relations.add('sdo0:identifier')
                golden_relations.add('geo:sfWithin')
            else:  # wijk
                granularity = """
                ?wijk sdo0:identifier "#wijkcode".
                ?wijk ^geo:sfWithin/^geo:sfWithin ?geb.
                """
                golden_relations.add('sdo0:identifier')
                golden_relations.add('geo:sfWithin')

            query = """
            SELECT DISTINCT %s
            WHERE {
                %s
                %s
                %s
                %s
            } %s
            """ % (return_keys, subject, building_part, granularity, filters, order)

        elif granularity_key in ('land'):
            return_keys = []

            # The subject is the same as before for other granularities
            if property_key == 'percelen':

                # First we determine whether we need the ?vbo block
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte' or year_filter_options or brt_property_key:
                    vbo = """
                    ?vbo a sor:Verblijfsobject;
                        sor:hoofdadres ?na;
                    """
                    golden_classes.add('sor:Verblijfsobject')
                    golden_relations.add('sor:hoofdadres')

                    # we determine whether we need the variable ?wo
                    if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                        wo_part = "sor:oppervlakte ?wo;"
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # we determine whether we need the variable ?geb
                    if year_filter_options:
                        vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb;'
                        golden_relations.add('sor:maaktDeelUitVan')
                    else:
                        vbo_geb_koppeling = ''

                    # If we don't need ?wo and don't need ?geb
                    # Then it seems we only need the vbo block in order to have brt_property_key filtering
                    # So we put the dot directly after ?na.

                    if wo_part == '' and vbo_geb_koppeling == '':
                        vbo = """
                        ?vbo a sor:Verblijfsobject;
                            sor:hoofdadres ?na.
                        """
                        golden_classes.add('sor:Verblijfsobject')
                        golden_relations.add('sor:hoofdadres')
                    elif wo_part != '' and vbo_geb_koppeling == '':
                        pass  # in this case the dot placement is already correct
                    elif wo_part == '' and vbo_geb_koppeling != '':
                        vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb.'  # we put the dot placement in the correct place
                else:
                    vbo = ''
                    vbo_geb_koppeling = ''
                    wo_part = ''

                # We retrieve ?bo if needed
                if year_filter_options:
                    bo_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # we determine if we need the ?na part at percelen block
                if wo_part != '' or vbo_geb_koppeling != '' or vbo != '':
                    per_na_part = 'sor:hoortBij ?na;'
                    golden_relations.add('sor:hoortBij')
                else:
                    per_na_part = ''

                # We determine whether we need ?po
                if key == '?po' or parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    po_part = 'sor:oppervlakte ?po;'
                    golden_relations.add('sor:oppervlakte')
                else:
                    po_part = ''

                # we determine whether we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                per = '?per a sor:Perceel;'
                # Now we determine the dot placement for the percelen block

                # If all the other statements not needed we put dot directly behind perceel
                if per_na_part == '' and po_part == '' and geo_part == '':
                    per = '?per a sor:Perceel.'
                    golden_classes.add('sor:Perceel')
                # change dot placement
                elif per_na_part != '' and po_part != '' and geo_part == '':
                    po_part = 'sor:oppervlakte ?po.'
                # change dot placement
                elif per_na_part != '' and po_part == '' and geo_part == '':
                    per_na_part = 'sor:hoortBij ?na.'
                # change dot placement
                elif per_na_part == '' and po_part != '' and geo_part == '':
                    po_part = 'sor:oppervlakte ?po.'
                # Note in all other cases the dot placement is already correct

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:hoofdadres ?na;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo.


                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.
                """
                '''

                subject = """
                %s   
                    %s
                    %s

                %s

                %s
                    %s
                    %s
                    %s
                """ % (vbo, vbo_geb_koppeling, wo_part, bo_part, per, per_na_part, po_part, geo_part)

                if amount_key is None:
                    return_keys.extend(['?per'])

                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?per) as ?aantal)')


            # The subject is the same as before for other granularities
            elif property_key == 'monumenten':

                # First we determine whether we need the ?vbo block
                # Note in some of these reasons we only need part of the vbo block to retrieve parcel information
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte' or brt_property_key or parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    # Note that when we need the vbo block we always need to link it to ?geb (monuments are gebouwen)
                    vbo = """
                    ?vbo a sor:Verblijfsobject;
                        sor:maaktDeelUitVan ?geb;
                    """
                    golden_classes.add('sor:Verblijfsobject')
                    golden_relations.add('sor:maaktDeelUitVan')

                    # We determine if we need ?na
                    # note we need ?na only when we eventually want to go to parcel information
                    if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                        vbo_na_part = 'sor:hoofdadres ?na;'
                        golden_relations.add('sor:hoofdadres')
                    else:
                        vbo_na_part = ''

                    # we determine whether we need the variable ?wo
                    if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                        wo_part = "sor:oppervlakte ?wo."
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # If both parts are empty, we put the dot directly after ?geb.
                    # In this case we only need this vbo block to perform a brt_filter_key filtering operation.
                    if vbo_na_part == '' and wo_part == '':
                        vbo = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb.
                        """
                    elif vbo_na_part != '' and wo_part == '':
                        vbo_na_part = 'sor:hoofdadres ?na.'
                    # Note in all other cases the dot placement is already correct
                else:
                    vbo = ''
                    vbo_na_part = ''
                    wo_part = ''

                # we instantiate the gebouwen part.
                geb = """
                ?geb a sor:Gebouw;
                    ^kad:gevestigdOp/kad:grondslag ?grondslag;
                """
                golden_classes.add('sor:Gebouw')
                golden_relations.add('kad:gevestigdOp')
                golden_relations.add('kad:grondslag')

                # we determine whether we need ?bo
                if year_filter_options or key == '?bo':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo;'
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # we determine whether we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # if we don't need these both we need to put a dot at the end of grondslag
                if bo_part == '' and geo_part == '':
                    geb = """
                    ?geb a sor:Gebouw;
                        ^kad:gevestigdOp/kad:grondslag ?grondslag.
                    """
                # determine dot placement in other cases
                elif bo_part != '' and geo_part == '':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo.'
                # note in all other cases the dot placement is already correct

                # We determine if we need the percelen block
                # The only reason currently we need parcels is when we need it for ?po
                # Note you might want to change this later if you want to retrieve only property IRI's and parcel IRI's
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                VALUES ?grondslag {
                    kad-con:GG 
                    kad-con:GWA 
                }

                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:hoofdadres ?na;
                    sor:oppervlakte ?wo.

                ?geb a sor:Gebouw;
                    ^kad:gevestigdOp/kad:grondslag ?grondslag;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per 
                    a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                subject = """
                VALUES ?grondslag {
                    kad-con:GG 
                    kad-con:GWA 
                }

                %s
                    %s
                    %s

                %s
                    %s
                    %s

                %s
                """ % (vbo, vbo_na_part, wo_part, geb, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])
                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            # The subject is the same as before for the other granularities
            elif property_key == 'huizen':
                # We instantiate the vbo part since we will always need it.
                # for now we leave a semicolon since there might be more parts needed
                vbo = """
                ?vbo a sor:Verblijfsobject;
                    sor:gebruiksdoel sor-con:woonfunctie;
                """
                golden_classes.add('sor:Verblijfsobject')
                golden_relations.add('sor:gebruiksdoel')

                # we determine whether we need to retrieve ?geb linked to our vbo's
                # this is needed when we need ?bo which can only be retrieved through ?geb
                if key == '?bo' or year_filter_options or extra_return_key == 'bouwjaar':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    vbo_geb_koppeling = ''

                # we determine whether we need ?wo
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                    wo_part = 'sor:oppervlakte ?wo;'
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # We determine whether we need ?na
                # note we only need ?na to connect our vbo's to parcel information
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte' or key == '?po':
                    vbo_na_part = 'sor:hoofdadres ?na;'
                    golden_relations.add('sor:hoofdadres')
                else:
                    vbo_na_part = ''

                # We determine whether we need the geometry:
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # Now we make sure the dot placement is correct
                # if all 3 are empty we put a dot at the beginning
                if vbo_geb_koppeling == '' and wo_part == '' and vbo_na_part == '' and geo_part == '':
                    vbo = """
                    ?vbo a sor:Verblijfsobject;
                        sor:gebruiksdoel sor-con:woonfunctie.
                    """

                # we treat all other cases that change the dot placement (other cases are ignored)
                elif vbo_geb_koppeling != '' and wo_part != '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling != '' and wo_part != '' and vbo_na_part == '' and geo_part == '':
                    wo_part = 'sor:oppervlakte ?wo.'
                elif vbo_geb_koppeling != '' and wo_part == '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part != '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part == '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part != '' and vbo_na_part == '' and geo_part == '':
                    wo_part = 'sor:oppervlakte ?wo.'
                elif vbo_geb_koppeling != '' and wo_part == '' and vbo_na_part == '' and geo_part == '':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb.'

                # We retrieve ?bo if it is needed by the query
                if key == '?bo' or year_filter_options or extra_return_key == 'bouwjaar':
                    bo_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # We retrieve parcel information if needed
                # The only reason currently we need parcels is when we need it for ?po
                # Note you might want to change this later if you want to retrieve only property IRI's and parcel IRI's
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:gebruiksdoel sor-con:woonfunctie;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.


                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''
                subject = """
                %s
                    %s
                    %s
                    %s
                    %s


                %s


                %s
                """ % (vbo, vbo_geb_koppeling, wo_part, vbo_na_part, geo_part, bo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')

            # the subject is the same as before for other granularities
            elif property_key == 'verblijfsobjecten':

                # We instantiate the vbo part since we will always need it.
                # for now we leave a semicolon since there might be more parts needed
                vbo = '?vbo a sor:Verblijfsobject;'
                golden_classes.add('sor:Verblijfsobject')

                # we determine whether we need to retrieve ?geb linked to our vbo's
                # this is needed when we need ?bo which can only be retrieved through ?geb
                if key == '?bo' or year_filter_options or extra_return_key == 'bouwjaar':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb;'
                    golden_relations.add('sor:maaktDeelUitVan')
                else:
                    vbo_geb_koppeling = ''

                # we determine whether we need ?wo
                if key == '?wo' or extra_return_key == 'oppervlakte' or house_filter_options:
                    wo_part = 'sor:oppervlakte ?wo;'
                    golden_relations.add('sor:oppervlakte')
                else:
                    wo_part = ''

                # We determine whether we need ?na
                # note we only need ?na to connect our vbo's to parcel information
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte' or key == '?po':
                    vbo_na_part = 'sor:hoofdadres ?na;'
                    golden_relations.add('sor:hoofdadres')
                else:
                    vbo_na_part = ''

                # We determine whether we need the geometry:
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry/geo:asWKT ?geo_wgs84.'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                else:
                    geo_part = ''

                # if all 4 parts are empty we put a dot directly after verblijfsobject
                if vbo_geb_koppeling == '' and wo_part == '' and vbo_na_part == '' and geo_part == '':
                    vbo = '?vbo a sor:Verblijfsobject.'
                # we treat all other cases that change the dot placement (other cases are ignored)
                elif vbo_geb_koppeling != '' and wo_part != '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling != '' and wo_part != '' and vbo_na_part == '' and geo_part == '':
                    wo_part = 'sor:oppervlakte ?wo.'
                elif vbo_geb_koppeling != '' and wo_part == '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part != '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part == '' and vbo_na_part != '' and geo_part == '':
                    vbo_na_part = 'sor:hoofdadres ?na.'
                elif vbo_geb_koppeling == '' and wo_part != '' and vbo_na_part == '' and geo_part == '':
                    wo_part = 'sor:oppervlakte ?wo.'
                elif vbo_geb_koppeling != '' and wo_part == '' and vbo_na_part == '' and geo_part == '':
                    vbo_geb_koppeling = 'sor:maaktDeelUitVan ?geb.'

                # We retrieve ?bo if it is needed by the query
                if key == '?bo' or year_filter_options or extra_return_key == 'bouwjaar':
                    bo_part = """
                    ?geb
                        sor:oorspronkelijkBouwjaar ?bo.
                    """
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # We retrieve parcel information if needed
                # The only reason currently we need parcels is when we need it for ?po
                # Note you might want to change this later if you want to retrieve only property IRI's and parcel IRI's
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:oppervlakte ?wo;
                    sor:hoofdadres ?na;
                    geo:hasGeometry/geo:asWKT ?geo_wgs84.

                ?geb
                    sor:oorspronkelijkBouwjaar ?bo.

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                subject = """
                %s
                    %s
                    %s
                    %s
                    %s

                %s

                %s
                """ % (vbo, vbo_geb_koppeling, wo_part, vbo_na_part, geo_part, bo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?vbo'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])
                else:
                    return_keys.append('(COUNT(DISTINCT ?vbo) as ?aantal)')

                # Same subject as before for other granularities
            else:  # 'gebouwen'

                # First we determine whether we need the ?vbo block
                # Note in some of these reasons we only need part of the vbo block to retrieve parcel information
                if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte' or brt_property_key or parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    # Note that when we need the vbo block we always need to link it to ?geb (monuments are gebouwen)
                    vbo = """
                    ?vbo a sor:Verblijfsobject;
                        sor:maaktDeelUitVan ?geb;
                    """
                    golden_classes.add('sor:Verblijfsobject')
                    golden_relations.add('sor:maaktDeelUitVan')

                    # We determine if we need ?na
                    # note we need ?na only when we eventually want to go to parcel information
                    if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                        vbo_na_part = 'sor:hoofdadres ?na;'
                        golden_relations.add('sor:hoofdadres')
                    else:
                        vbo_na_part = ''

                    # we determine whether we need the variable ?wo
                    if house_filter_options or key == '?wo' or extra_return_key == 'oppervlakte':
                        wo_part = "sor:oppervlakte ?wo."
                        golden_relations.add('sor:oppervlakte')
                    else:
                        wo_part = ''

                    # If both parts are empty, we put the dot directly after ?geb.
                    # In this case we only need this vbo block to perform a brt_filter_key filtering operation.
                    if vbo_na_part == '' and wo_part == '':
                        vbo = """
                        ?vbo a sor:Verblijfsobject;
                            sor:maaktDeelUitVan ?geb.
                        """
                    elif vbo_na_part != '' and wo_part == '':
                        vbo_na_part = 'sor:hoofdadres ?na.'
                    # Note in all other cases the dot placement is already correct
                else:
                    vbo = ''
                    vbo_na_part = ''
                    wo_part = ''

                golden_classes.add('sor:Gebouw')
                # we instantiate the gebouwen part.
                geb = '?geb a sor:Gebouw;'

                # we determine whether we need ?bo
                if year_filter_options or key == '?bo':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo;'
                    golden_relations.add('sor:oorspronkelijkBouwjaar')
                else:
                    bo_part = ''

                # we determine whether we need geometry
                if extra_return_key == 'geometrie':
                    geo_part = 'geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].'
                    golden_relations.add('geo:hasGeometry')
                    golden_relations.add('geo:asWKT')
                    golden_relations.add('rdfs:isDefinedBy')
                else:
                    geo_part = ''

                # if we don't need these both we need to put a dot at the end of grondslag
                if bo_part == '' and geo_part == '':
                    geb = '?geb a sor:Gebouw.'
                # determine dot placement in other cases
                elif bo_part != '' and geo_part == '':
                    bo_part = 'sor:oorspronkelijkBouwjaar ?bo.'
                # note in all other cases the dot placement is already correct

                # We determine if we need the percelen block
                # The only reason currently we need parcels is when we need it for ?po
                # Note you might want to change this later if you want to retrieve only property IRI's and parcel IRI's
                if parcel_filter_options or extra_return_key == 'perceeloppervlakte':
                    per_part = """
                    OPTIONAL { 
                    ?per a sor:Perceel;
                        sor:hoortBij ?na;
                        sor:oppervlakte ?po.
                    }
                    """
                    golden_classes.add('sor:Perceel')
                    golden_relations.add('sor:hoortBij')
                    golden_relations.add('sor:oppervlakte')
                else:
                    per_part = ''

                '''
                subject = """
                ?vbo a sor:Verblijfsobject;
                    sor:maaktDeelUitVan ?geb;
                    sor:hoofdadres ?na;
                    sor:oppervlakte ?wo.

                ?geb a sor:Gebouw;
                    sor:oorspronkelijkBouwjaar ?bo;
                    geo:hasGeometry [ geo:asWKT ?geo_wgs84; rdfs:isDefinedBy bag: ].

                OPTIONAL { ?per a sor:Perceel;
                    sor:hoortBij ?na;
                    sor:oppervlakte ?po. }
                """
                '''

                subject = """
                %s
                    %s
                    %s

                %s
                    %s
                    %s

                %s
                """ % (vbo, vbo_na_part, wo_part, geb, bo_part, geo_part, per_part)

                if amount_key is None:
                    return_keys.extend(['?geb'])

                    if extra_return_key == 'oppervlakte':
                        return_keys.extend(['?wo'])
                    if extra_return_key == 'perceeloppervlakte':
                        return_keys.extend(['?po'])
                    if extra_return_key == 'bouwjaar':
                        return_keys.extend(['?bo'])
                    if extra_return_key == 'geometrie':
                        return_keys.extend(['?geo_wgs84'])


                else:
                    return_keys.append('(COUNT(DISTINCT ?geb) as ?aantal)')

            return_keys = ' '.join(return_keys)

            query = """
            SELECT DISTINCT %s
            WHERE {
                %s
                %s
                %s
            } %s
            """ % (return_keys, subject, building_part, filters, order)


        else:
            query = None

        '''
        Depending on whether we have a land granularity we do something different here. 

        If we do not have land granularity we also put the adress in our answer with seperation tokens
        '''

        '''
        UNCOMMNENT THIS PART IF YOU WANT TO HAVE STANDARD FUNCTIONALITY OF THIS METHOD:

        # If granularity_key == land we simply format the query into a single line
        if granularity_key == 'land':
            answer = self.format_query(query)

        # In the case we don't have land granularity we also put the adress in our answer with seperator tokens
        else:
            answer = Functions.generate_function_text(
                SpecialTokens.LOOKUP_ADDRESS, address, self.format_query(query)
            )

        '''
        # This line is temporary to analyze the queries:
        #answer = query

        if granularity_key == 'land':
            answer = self.format_query(query)
        else:
            answer = Functions.generate_function_text(
                SpecialTokens.LOOKUP_ADDRESS, address, self.format_query(query)
            )
        return answer, list(golden_classes), list(golden_relations)

    def update_house_filter_value(self, property_key, template_words, index):
        house_filter_option, filter_synonym = self.get_key_value(self.size_filter_synonyms)
        house_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))

        probability_mask = [None, 100]
        classes = [HOUSE_SURFACE + (SURFACE if property_key != 'percelen' else []), [filter_synonym]]

        choice = random.randint(0, 2)
        if choice == 2:
            classes.extend([[house_filter_value], SQUARE_METER])
            probability_mask.extend([100, None])
        elif choice == 1:
            classes.append([house_filter_value + random.choice(tuple(SQUARE_METER))])
            probability_mask.append(100)
        else:
            classes.append([house_filter_value])
            probability_mask.append(100)

        filter_synonym = self.get_text_by_classes(classes, probability_mask=probability_mask)

        self.update_tokens(template_words, None, get_template_formatted(Template.HOUSE_SIZE.value),
                           filter_synonym, index=index)

        return house_filter_option, house_filter_value

    '''
    This is the main function that gets used in query_registry.py

    This function behaves different based on whether related = None

    1) If related is NOT None
    This means someone has pressed a button in the chatbot related to one of the brt_properties.
    So basically a follow up question with a new brt_property is asked related to the previous question.

    We see in the code that in this case we unpack the related tuple to know which new brt_property we will
    answer the previous related question about.

    We retrieve the last answer_kwargs from the stack.

    We check whether we already have adress_data in our answer_kwargs, if it is not there we return:
    answer = "Wat is het adres waar u in geïnteresseerd bent?"

    If there is adress_data in the answer_kwargs:
    We get either get granularity_key and some randomly selected option,
    amount_key and string 'aantal',
    property_key and some randomly selected property.

    we change whatever we retrieved in our answer_kwargs and again get a new answer & answer_kwargs
    We do not return a question since in query_registry we simply take the randomly selected related_key
    as a question.

    2) If related IS None
    Then a new question entirely is generated.

    In this case we first run the function get_question_tokens(). 
    This function selects a random template.
    template_words: list with words in the template (non-placeholders typos introduces)
    token_to_indices: maps words in the list to their indices
    attributes: the dictionary related to the question/template in the YAML file



    '''

    def get_question_answer(self, existing_dependencies, answer_kwargs_stack, related, *args, **kwargs):
        if related is not None:
            key, value = related

            answer_kwargs = self.get_last_answer_kwargs(answer_kwargs_stack)
            if 'address_data' not in answer_kwargs:
                answer = get_dependency_text(Dependency.ADDRESS)
            else:
                answer_kwargs[key] = value
                answer = self.get_answer(**answer_kwargs)

            return None, answer, answer_kwargs

        template_words, token_to_indices, attributes = self.get_question_tokens()

        # We check if our specific template requires us to return more than the standard variables
        if 'extra_return' in attributes.keys():
            extra_return_key = attributes['extra_return']
        else:
            extra_return_key = None

        '''
        Here we start checking whether the placeholders are relevant for a certain template (we do this for
        all the possible placeholders).

        - If we have {property} is in our template
        We select a random key from self.property_synonyms and a random synonym belonging to it.
        Then we use update_tokens() function. This function basically searches for the placeholder in our 
        dictionary (that maps tokens to indices),  then at the found indices we change the placeholder to
        the synonym.

        If it is in attribute keys but not as a placeholder in the natural language question (sometimes we have this)
        Then we set property_key variable to whatever is there in the YAML file.

        If it is nowhere we set property_key to None.

        - If we have {amount}, {brt_property}, ...., {order} in our selected template we do a similar thing.

        -IMPORTANT: For {Granularity} when we do the above procedure if it is not there as a placeholder
        and neither as an attribute we automatically set granularity_key = 'land'



        '''

        if get_template_formatted(Template.PROPERTY.value) in token_to_indices:
            # get_key_value() returns a random key and a random synonym belonging to that key

            # We create a temporary dictionary with property synonyms where we exclude the excluded property
            # from our dictionary if such restrictions should be applied
            property_synonyms_exclusions_applied = self.property_synonyms
            if 'excludeproperty' in attributes.keys():
                property_synonyms_exclusions_applied = {key: value for key, value in
                                                        property_synonyms_exclusions_applied.items() if
                                                        key not in attributes['excludeproperty']}

            # We create a temporary dictionary if the attributes force the property to be a certain value
            # this temporary dictionary reflects the desired restrictions
            if 'property' in attributes.keys():
                property_synonyms_exclusions_applied = {key: value for key, value in
                                                        property_synonyms_exclusions_applied.items() if
                                                        key == attributes['property']}

            property_key, property_synonym = self.get_key_value(property_synonyms_exclusions_applied)

            self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.PROPERTY.value),
                               property_synonym)
        elif Template.PROPERTY.value in attributes.keys():
            property_key = attributes[Template.PROPERTY.value]
        else:
            property_key = None

        if get_template_formatted(Template.AMOUNT.value) in token_to_indices:
            amount_key, amount_synonym = 'aantal', random.choice(AMOUNT)

            self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.AMOUNT.value),
                               amount_synonym)
        elif Template.AMOUNT.value in attributes.keys():
            amount_key = attributes[Template.AMOUNT.value]
        else:
            amount_key = None

        if get_template_formatted(Template.BRT_PROPERTY.value) in token_to_indices:
            brt_property_key, brt_property_synonym = self.get_key_value(brt_properties)

            self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.BRT_PROPERTY.value),
                               brt_property_synonym)
        elif Template.BRT_PROPERTY.value in attributes.keys():
            brt_property_key = attributes[Template.BRT_PROPERTY.value]
        else:
            brt_property_key = None

        if get_template_formatted(Template.GRANULARITY.value) in token_to_indices:
            granularity_synonyms_exclusions_applied = self.granularity_synonyms
            if 'excludegranularity' in attributes.keys():
                granularity_synonyms_exclusions_applied = {key: value for key, value in
                                                        granularity_synonyms_exclusions_applied.items() if
                                                        key not in attributes['excludegranularity']}
            granularity_key, granularity_synonym = self.get_key_value(granularity_synonyms_exclusions_applied)

            self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.GRANULARITY.value),
                               granularity_synonym)
        elif Template.GRANULARITY.value in attributes.keys():
            granularity_key = attributes[Template.GRANULARITY.value]
        else:
            granularity_key = 'land'

        # HERE WE PUT SOME RESTRICTIONS FOR DEBUGGING ONLY:
        # if property_key != 'huizen' or (granularity_key not in ('wijk', 'buurt', 'gemeente', 'provincie')):
        #    return None, None, None

        if get_template_formatted(Template.ORDER.value) in token_to_indices:
            order_synonyms_exclusions_applied = self.order_synonyms
            if 'excludeorder' in attributes.keys():
                order_synonyms_exclusions_applied = {key: value for key, value in
                                                        order_synonyms_exclusions_applied.items() if
                                                        key not in attributes['excludeorder']}

            order_key, order_synonym = self.get_key_value(order_synonyms_exclusions_applied)
            self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.ORDER.value),
                               order_synonym)
        elif Template.ORDER.value in attributes.keys():
            order_key = attributes[Template.ORDER.value]
        else:
            order_key = None

        '''
        If granularity_key = 'land'
        We check whether there is a {location} placeholder in the template.
        In this case we remove the {location} placeholder. (replace it with '')

        If granularity_key is something else, we also check whether there is a {location} placeholder.
        If there is, we generate a location (without street, number and it is not a postal code version)
        It returns a dictionary and also a combined string.
        We replace the {location} placeholder with our generated location.
        Note that if there is no {location} placeholder we check whether it is in the attributes, and
        change our adress_data to that.
        If it is neither there as placeholder nor in the attributes we set adress_data as None.


        '''

        def get_location_tailored_to_granularity(address_str, granularity):
            '''
            Note that in our current setup we extract cities from our locations file. This does not make sense
            if the granularity we are working with is province or municipality instead.

            Therefore I do a call to the location server to extract the names of the municipality or province
            to which the city belongs if we have such a granularity.



            :param address_str: the city name
            :param granularity: the granularity which our question considers
            :return: the correct name of the granularity we are considering to format in a natural language question
            '''

            SUGGEST_URL = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
                          '?fq=type:(gemeente OR woonplaats OR adres OR provincie) AND bron:BAG&q={}'

            def get_address(address):
                url = SUGGEST_URL.format(address)
                address = requests.get(url).json()['response']['docs'][0]

                return address

            if granularity == 'provincie':
                address_dictionary = get_address(address_str)
                return address_dictionary['provincienaam']

            elif granularity == 'gemeente':
                address_dictionary = get_address(address_str)
                return address_dictionary['gemeentenaam']
            else:
                return address_str


        if granularity_key == 'land':
            address_data = {}
            if get_template_formatted(Template.LOCATION.value) in token_to_indices:
                self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.LOCATION.value),
                                   '')
        else:
            if get_template_formatted(Template.LOCATION.value) in token_to_indices:
                address_data, address_str = self.get_address(street=False, number=False, generate_postal_code=False)
                address_str = get_location_tailored_to_granularity(address_str,granularity_key)
                self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.LOCATION.value),
                                   address_str)
            elif Template.LOCATION.value in attributes.keys():
                address_data = attributes[Template.LOCATION.value]
            else:
                address_data = None

        '''
        Next we search for year_filter_options.

        FIRST WE CHECK WHETHER constructionyear IS IN THE ATTRIBUTES:
        -If it is, we set year_filter_options whatever is there in the attributes within the YAML file about
        constructionyear. ( needed because it is an interval in this case, bigger than first, smaller than second)
        Then we look for {constructionyear} in the template (perhaps multiple locations)
        Then we fill them with randomly generated numbers (strings) to replace the placeholders.
        This is especially important when working with an interval. The year_filter_options for example will
        then allow us to put two filters to obtain the interval filter. This is often when it is in attributes.

        -If constructionyear is not in attributes will select year_filter_options at random from self.year_filter_synonyms 
        We create some string including a filter_option_synonym and a value. We replace the placeholder with that now
        '''
        year_filter_options, year_filter_values = [], []
        # checking if constructionyear is in the attributes of the YAML file
        if Template.YEAR_FILTER.value in attributes.keys():
            year_filter_options = attributes[Template.YEAR_FILTER.value]

            if get_template_formatted(Template.YEAR_FILTER.value) in token_to_indices:
                for index in token_to_indices[get_template_formatted(Template.YEAR_FILTER.value)]:
                    # we generate digit string of 1-4 character ( we loop so each placeholder gets different value)
                    # We fill the placeholders
                    year_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))
                    self.update_tokens(template_words, None, get_template_formatted(Template.YEAR_FILTER.value),
                                       year_filter_value, index=index)
                    year_filter_values.append(year_filter_value)
        elif get_template_formatted(Template.YEAR_FILTER.value) in token_to_indices:
            for index in token_to_indices[get_template_formatted(Template.YEAR_FILTER.value)]:
                year_filter_option, filter_synonym = self.get_key_value(self.year_filter_synonyms)
                year_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))
                # with 50% probability we select a buildyear. We definitely select a filter synonym.
                # we also definitely select some year_filter_value by using get_text_by_classes()
                filter_synonym = self.get_text_by_classes([BUILD_YEAR, [filter_synonym], [year_filter_value]],
                                                          probability_mask=[100, 100, 100])

                self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.YEAR_FILTER.value),
                                   filter_synonym, index=index)

                year_filter_options.append(year_filter_option)
                year_filter_values.append(year_filter_value)

        '''
        We do something similar as above for parcel_filter_options & values.
        '''
        parcel_filter_options, parcel_filter_values = [], []
        if Template.PARCEL_SIZE.value in attributes.keys():
            parcel_filter_options = attributes[Template.PARCEL_SIZE.value]

            if get_template_formatted(Template.PARCEL_SIZE.value) in token_to_indices:
                for index in token_to_indices[get_template_formatted(Template.PARCEL_SIZE.value)]:
                    parcel_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))
                    self.update_tokens(template_words, None, get_template_formatted(Template.PARCEL_SIZE.value),
                                       parcel_filter_value, index=index)
                    parcel_filter_values.append(parcel_filter_value)
        elif get_template_formatted(Template.PARCEL_SIZE.value) in token_to_indices:
            for index in token_to_indices[get_template_formatted(Template.PARCEL_SIZE.value)]:
                parcel_filter_option, filter_synonym = self.get_key_value(self.size_filter_synonyms)
                parcel_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))

                probability_mask = [100, 100]
                # We have a list of two lists here.
                # The first list might be a concatenation of PARCEL_SURFACE and SURFACE
                # (based on what property_key we have)
                classes = [PARCEL_SURFACE + (SURFACE if property_key == 'percelen' else []), [filter_synonym]]

                # Here we randomly decide how to extend our classes and probability mask.
                choice = random.randint(0, 2)
                if choice == 2:
                    # 2 lists added
                    classes.extend([[parcel_filter_value], SQUARE_METER])
                    probability_mask.extend([100, None])
                elif choice == 1:
                    # one list added with a probability mask (concatenated)
                    classes.append([parcel_filter_value + random.choice(tuple(SQUARE_METER))])
                    probability_mask.append(100)
                else:
                    # one list added (NOT concatenated)
                    classes.append([parcel_filter_value])
                    probability_mask.append(100)
                filter_synonym = self.get_text_by_classes(classes, probability_mask=probability_mask)

                self.update_tokens(template_words, token_to_indices, get_template_formatted(Template.PARCEL_SIZE.value),
                                   filter_synonym, index=index)

                parcel_filter_options.append(parcel_filter_option)
                parcel_filter_values.append(parcel_filter_value)

        '''
        We do something similar for filters on house size.
        '''
        house_filter_options, house_filter_values = [], []
        if Template.HOUSE_SIZE.value in attributes.keys():
            house_filter_options = attributes[Template.HOUSE_SIZE.value]

            if get_template_formatted(Template.HOUSE_SIZE.value) in token_to_indices:
                for index in token_to_indices[get_template_formatted(Template.HOUSE_SIZE.value)]:
                    house_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))
                    self.update_tokens(template_words, None, get_template_formatted(Template.HOUSE_SIZE.value),
                                       house_filter_value, index=index)
                    house_filter_values.append(house_filter_value)

        elif get_template_formatted(Template.HOUSE_SIZE.value) in token_to_indices:
            for index in token_to_indices[get_template_formatted(Template.HOUSE_SIZE.value)]:
                house_filter_option, filter_synonym = self.get_key_value(self.size_filter_synonyms)
                house_filter_value = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 4)))

                probability_mask = [100, 100]
                classes = [HOUSE_SURFACE, [filter_synonym]]

                choice = random.randint(0, 2)
                if choice == 2:
                    classes.extend([[house_filter_value], SQUARE_METER])
                    probability_mask.extend([100, None])
                elif choice == 1:
                    classes.append([house_filter_value + random.choice(tuple(SQUARE_METER))])
                    probability_mask.append(100)
                else:
                    classes.append([house_filter_value])
                    probability_mask.append(100)

                # Note get_text_by_classes simply returns a string like this.
                filter_synonym = self.get_text_by_classes(classes, probability_mask=probability_mask)

                self.update_tokens(template_words, None, get_template_formatted(Template.HOUSE_SIZE.value),
                                   filter_synonym, index=index)

                house_filter_options.append(house_filter_option)
                house_filter_values.append(house_filter_value)

        '''
        Here we generate answer_kwargs based on all the options and values we had before.
        '''
        answer_kwargs = dict(
            order_key=order_key,
            property_key=property_key,
            granularity_key=granularity_key,
            brt_property_key=brt_property_key, amount_key=amount_key,
            year_filter_options=year_filter_options, year_filter_values=year_filter_values,
            house_filter_options=house_filter_options, house_filter_values=house_filter_values,
            parcel_filter_options=parcel_filter_options, parcel_filter_values=parcel_filter_values,
            extra_return_key=extra_return_key,
        )

        '''
        Here we deal with the adresses.

        QUESTION: What is own_adress???

        In the last case in the if-statements we return an adress question (what is your adress?)

        We concatenate the list with question words into a sentence.



        '''
        if address_data is not None:
            answer_kwargs.update({
                'own_address': False,
                'address_data': address_data,
            })
            answer, golden_classes, golden_relations = self.get_answer(**answer_kwargs)
        elif Dependency.ADDRESS in existing_dependencies.keys():
            answer_kwargs.update({
                'own_address': True,
                'address_data': existing_dependencies[Dependency.ADDRESS],
            })
            answer, golden_classes, golden_relations = self.get_answer(**answer_kwargs)
        else:
            answer = get_dependency_text(Dependency.ADDRESS)

        question = self.tokens_to_question(template_words)



        return question, answer, answer_kwargs, golden_classes, golden_relations
