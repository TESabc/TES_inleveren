from enum import Enum


class SpecialTokens(Enum):
    RELATED = '<rel>'
    INTERACTION = '<i>'
    SEPARATOR = '<sep>'
    QA = '<qa>'
    ADDRESS = '<address>'
    LOOKUP_ADDRESS = '<lookup_adres>'


class TimePeriod(Enum):
    TODAY = frozenset({'vandaag', 'deze dag'})
    YESTERDAY = frozenset({'gisteren', 'gister', 'een dag terug', 'een dag geleden'})
    DAY_BEFORE_YESTERDAY = frozenset({'eergisteren', 'eergister', 'twee dagen terug', 'twee dagen geleden'})
    WEEK = frozenset({'deze week', 'week', 'de huidige week'})
    LAST_WEEK = frozenset({'afgelopen week', 'verleden week', 'vorige week', 'laatste week'})
    MONTH = frozenset({'deze maand', 'maand', 'de huidige maand'})
    LAST_MONTH = frozenset({'afgelopen maand', 'verleden maand', 'vorige maand', 'laatste maand'})
    YEAR = frozenset({'dit jaar', 'jaar', 'het huidige jaar'})
    LAST_YEAR = frozenset({'afgelopen jaar', 'verleden jaar', 'vorige jaar', 'laatste jaar'})


class Transportation(Enum):
    CAR = frozenset({'auto', 'automobiel', 'bak', 'bolide', 'brik', 'vehikel', 'wagen'})
    BIKE = frozenset({'fietsen', 'rijwiel', 'stalen ros', 'tweewieler', 'fiets'})
    WALK = frozenset({'lopen', 'lopen', 'lopend', 'te voet', 'voet'})


class Time(Enum):
    MINUTES = frozenset({'minuten', 'minuut'})


class Status(Enum):
    BOUW_GESTART = frozenset({'bouw gestart', 'bouw begonnen', 'status gestart'})
    BOUWVERGUNNING_VERLEEND = frozenset({'bouwvergunning verleend', 'status vergunning verleend'})
    NIET_GEREALISEERD_PAND = frozenset({'niet gerealiseerd pand', 'ongerealiseerd pand', 'status ongerealiseerd'})
    PAND_BUITEN_GEBRUIK = frozenset({'pand buiten gebruik', 'pand niet in gebruik', 'status buiten gebruik'})
    PAND_GESLOOPT = frozenset({'pand gesloopt', 'gesloopt pand', 'pand verwoest', 'status gesloopt', 'gesloopt'})
    PAND_IN_GEBRUIK = frozenset({'pand in gebruik', 'gebruikt pand', 'status gebruik', 'in gebruik'})
    PAND_IN_GEBRUIK_NIET_INGEMETEN = frozenset({'pand in gebruik (niet ingemeten)', 'niet ingemeten gebruikt pand',
                                                'status gebruik niet ingemeten'})
    SLOOPVERGUNNING_VERLEEND = frozenset({'sloopvergunning verleend', 'pand met sloopvergunning', 'status sloop'})
    PAND_TEN_ONRECHTE_OPGEVOERD = frozenset({'pand ten onrechte opgevoerd', 'status ten onrechte opgevoerd',
                                             'pand the onrecht'})
    VERBOUWING_PAND = frozenset({'verbouwing pand', 'panden in verbouwing', 'status verbouwd', 'in verbouwing'})


class Permit(Enum):
    EVENT_PERMIT = ['evenementen', 'evenementen vergunning', 'evenementvergunning']
    ENVIRONMENT_PERMIT = ['milieu', 'milieu vergunning', 'milieuvergunning']
    FELLING_PERMIT = ['kap', 'kapvergunning', 'hakvergunning', 'houwvergunning']
    BUILD_PERMIT = ['bouw', 'bouwvergunning']
    EXPLOITATION_PERMIT = ['exploitatie', 'exploitatievergunning']
    ENVIRONMENT_ZONING_PERMIT = ['omgeving', 'omgevingsplan', 'omgevingsvergunning']
    DESTINATION_PERMIT = ['bestemming', 'bestemmingsplan', 'bestemmingsvergunning']
    ORDINANCES_REGULATIONS_PERMIT = ['verordeningen', 'regelementen', 'verordeningen en regelementen']


WIJK = ['wijk']
WIJKEN = ['wijken']
BUURT = ['buurt']
BUURTEN = ['buurten']
GEMEENTE = ['gemeente']
GEMEENTEN = ['gemeenten']
PROVINCIE = ['provincie']
LAND = ['land', 'nederland']


class Granularity(Enum):
    WIJK = WIJK
    BUURT = BUURT
    GEMEENTE = GEMEENTE
    PROVINCIE = PROVINCIE


UNKNOWN_QUESTION = "Ik heb geen data om deze vraag te beantwoorden. Heeft u een andere vraag?"
MORE_INFORMATION = "Ik heb extra informatie nodig om deze vraag te beantwoorden. Bent u in geïnteresseerd in {}?"
ADDRESS_QUESTION = "Wat is het adres waar u in geïnteresseerd bent?"
CHANGE_ADDRESS_SUGGESTION = 'ander adres'

AKTE = [
    'akte', 'aktes', 'akten', 'aktedelen', 'stuk', 'stukken', 'stukdelen'
]

SNELWEG = [
    'snelweg', 'autobaan', 'autosnelweg', 'autostrade'
]

SPOORWEG = [
    'spoorweg', 'spoorwegovergang',
]

AKTE_GROUPS = [
    'groep stuk', 'groepen stukken', 'groepen stukdelen',
    'groep akten', 'groepen aktes', 'groepen akten', 'overzicht van stukken', 'overzicht van de aktes',
    'overzicht van de stukken', 'overzicht van alle akten',
]

HOUSE_PRICE = [
    'gemiddelde woning waarde', 'woning waarde gemiddeld', 'gemiddelde woningwaarde',
    'gemiddeldewoningwaarde', 'prijsindicatie', 'gemiddelde woz waarde'
]

DISTURBANCE = [
    'gemiddelde overlast', 'geluidsoverlast', 'overlast', 'herrie',
]

CRIMINALITY = [
    'criminaliteit', 'criminaliteit cijfer', 'misdrijven', 'veilig', 'veiligheid', 'misdaad',
    'misdaad cijfers', 'gemiddelde aantal misdrijven'
]

LIVING_COST = [
    'woonkosten', 'woon kosten', 'belasting', 'belastingen', 'leef kosten', 'woon belasting', 'gemiddelde woonkosten'
]

POPULATION_DENSITY = [
    'dichtbevolktheid', 'dichtbevolkt', 'bevolkingsdichtheid'
]

PRICE_INDICATION = [
    'prijsindicatie', 'kostprijs', 'indicatie prijs'
]

OVERVIEW = [
    'overzicht', 'dashboard'
]

INSPECTION = [
    'schouw', 'papieren schouw', 'papierenschouw'
]

HOUSES = [
    'huizen', 'huis', 'woning', 'woningen'
]

PREMISES = [
    'gebouwen', 'gebouw',  'panden', 'pand',
]

ACCOMODATION_OBJECTS = [
    'verblijfsobject', 'verblijfsobjecten',
]

RENTAL_PROPERTIES = [
    'huurwoningen', 'woningen te huur'
]

OWNER_OCCUPIED_PROPERTIES = [
    'koopwoningen', 'woningen te koop'
]

URBANITY = [
    'stedelijkheid', 'mate van stedelijkheid',
]

ENERGY_CONSUMPTION = [
    'gemiddelde energieverbruik', 'energieverbruik'
]

HOUSE_TYPES = [
    'woningtypen', 'woning typen', 'typen woningen', 'woningtypes', 'woning types', 'types woningen'
]

RADIUS = [
    'straal', 'radius'
]

OLDER = [
    'kleiner', 'kleiner dan', 'ouder dan', 'ouder', 'voor', 'van voor'
]

EQUAL = [
    'gelijk', 'gelijk aan', 'van', 'is'
]

NEWER = [
    'groter', 'groter dan', 'jonger dan', 'jonger', 'nieuwer', 'nieuwer dan', 'na', 'van na'
]

BEFORE = [
    'kleiner', 'kleiner dan', 'ouder dan', 'ouder', 'voor', 'van voor'
]

AFTER = [
    'groter', 'groter dan', 'nieuwer', 'nieuwer dan', 'na', 'van na'
]

BETWEEN = [
    'tussen'
]

SMALLER = [
    'kleiner', 'kleiner dan', 'minder', 'minder dan'
]

BIGGER = [
    'groter', 'groter dan', 'meer', 'meer dan'
]

SQUARE_METER = [
    'vierkante meter', 'meter', 'm2', 'm'
]

# verwijderd: 'in gebouwd'
BUILD_YEAR = [
    'gebouw bouwjaar', 'bouwjaar', 'gebouwd', 'gebouwd in', 'jaar', 'ontstaansjaar'
]

STATUS = [
    'status', 'hoedanigheid', 'staat', 'toestand'
]

GEBRUIKSDOEL = [
    'gebruiksdoel'
]

SURFACE = [
    'oppervlakte', 'oppervlak', 'bovenste vlak', 'buitenste gedeelte'
]

# wegehaald: 'woning bovenste vlak', 'woning buitenste gedeelte','huis', 'woning',  'huis afmetingen', 'huisformaat', 'woningformaat',
HOUSE_SURFACE = [
    'woning oppervlakte', 'huis oppervlak',
    'woningoppervlakte', 'vloer oppervlakte', 'vloeroppervlakte',
    'woonoppervlakte', 'woon oppervlakte '
]

PARCEL = [
    'perceel', 'percelen', 'tuin', 'tuinen', 'kavel'
]

STREETS = [
    'straten', 'straat'
]

# Deze verwijderd: , 'tuin', 'perceel', 'kavel', 'tuin formaat',
#     'tuin afmetingen', 'tuin oppervlakte'
PARCEL_SURFACE = [
    'perceeloppervlakte', 'perceel oppervlakte', 'perceel oppervlak'
]

PARCEL_NUMBER = [
    'perceel nummer', 'perceel nr', 'nummer', 'perceelnummer'
]

LOCATION = [
    'locatie', 'adres'
]

CADASTRAL_MAP = [
    'kaart', 'kadastrale kaart', 'kadaster kaart', 'grenzen', 'kadastrale grenzen'
]

TOPO_TIJDREIS = [
    'topotijdreis', 'tijdreis', 'topo tijdreis'
]

HOOGTE = [
    'hoog', 'hoogte', 'hoge'
]

OWNER = frozenset({
    'eigenaar', 'bezitter'
})

OWNERSHIP_INFORMATION = frozenset({
    'eigendomsinformatie', 'eigendom informatie', 'informatie over eigendom', 'verkoop prijs', 'verkoopsprijs',
    'laatste verkoops prijs'
})

PURCHASE_PRICE_INFORMATION = frozenset({
    'koopsom informatie', 'koopsominformatie', 'verkoopsprijzen', 'koopsom'
})

HOUSE_REPORT = frozenset({
    'woning rapport', 'woningrapport', 'wooninformatie'
})

MONUMENTAL_STATUS = frozenset({
    'monument', 'gebouwen met monumentale status'
})

MORTGAGE_INFORMATION = frozenset({
    'hypotheek informatie', 'hypotheekinformatie', 'informatie over hypotheek', 'beslag', 'beslag gelegd',
    'beslag leggen'
})

NEAREST = frozenset({
    'dichstbijzijnde', 'dichtst bij', 'nabij'
})

CREDITOR = frozenset({
    'schuldeiser', 'crediteur'
})

LEGACY = frozenset({
    'nalatenschap', 'erven', 'ervenis'
})

BOUNDARY_SETTING = frozenset({
    'grensvaststelling', 'vaststelling kadastrale grens'
})

VARIANT = [
    'variant', 'type', 'eigenschap'
]

ANNOUNCEMENT = [
    'lokale bekendmakingen', 'bekendmakingen', 'officiële bekendmakingen'
]

INHABITANTS = [
    'inwoners', 'bewoners', 'burger'
]

HOUSHOLD_COMPOSITION = [
    'kinderen', 'kindvriendelijk', 'huishoudens', 'huishouden samenstelling'
]

HOUSEHOLDS = [
    'huishoudens', 'familie', 'gezin', 'huisgezin', 'huishouding'
]

AVERAGE_HOUSEHOLD_SIZE = [
    'gemiddelde huishouden grootte', 'huishouden grootte', 'huishouden formaat'
]

GENDER = [
    'geslacht'
]

AGE_GROUPS = [
    'leeftijdsgroepen', 'leeftijdscategorieen', 'leeftijd', 'jongeren', 'kinderen', 'jongen kinderen'
]

MARITAL_STATUS = [
    'burgerlijke staat', 'huwelijkse staat'
]

ORIGIN = [
    'herkomst', 'afkomst', 'oorsprong'
]

DENSITY = [
    'bouwdichtheid', 'mate van stedelijkheid', 'dichtbevolktheid'
]

brt_properties = {
    'fort': ['fort', 'forten', 'vesting'],
    'gemaal': ['gemaal', 'watergemaal', 'gemalen', 'afwateringsinstallatie'],
    'gemeentehuis': ['gemeentehuis', 'gemeentehuizen', 'raadhuis', 'stadhuis'],
    'gevangenis': ['gevangenis', 'bajes', 'bak', 'cachot', 'cel', 'gevang', 'gevangenhuis', 'gribus', 'huis', 'van',
                   'bewaring', 'kast', 'kerker', 'kot', 'lik', 'nor', 'penitentiaire', 'inrichting', 'pensionaat',
                   'petoet', 'pot', 'rijkshotel', 'strafgevangenis', 'strafinrichting', 'arrestantenlokaal'],
    'huizenblok': ['huizenblok', 'huizenblokken'],
    'kapel': ['kapel', 'kapels', 'kapellen', 'bedehuisje', 'bedeplaats', 'bedeplaatsen'],
    'kasteel': ['kasteel', 'kastelen', 'burcht', 'chateau', 'slot', 'vesting'],
    'kerk': ['kerk', 'kerken', 'basiliek', 'bedehuis', 'godshuis', 'kathedraal', 'kerkgebouw', 'tempel'],
    'koeltoren': ['koeltoren', 'koeltorens'],
    'kunstijsbaan': ['kunstijsbaan', 'kunstijsbanen'],
    'lichttoren': ['lichttoren', 'lichttorens'],
    'luchtwachttoren': ['luchtwachttoren', 'luchtwachttorens'],
    'manege': ['manege', 'maneges', 'paardenrijschool', 'paardenrijverblijf'],
    'moskee': ['moskee', 'moskeeën', 'moskeeen'],
    'observatorium': ['observatorium', 'observatoria', 'sterrenwacht', 'sterrenwachten'],
    'paleis': ['paleis', 'paleizen'],
    'parkeerdak, parkeerdek, parkeergarage': ['parkeerdak', 'parkeerdaken', 'parkeerdek', 'parkeergarages',
                                              'parkeergarages', 'parkeerdak, parkeerdek, parkeergarage'],
    'politiebureau': ['politiebureau', 'politiebureaus', 'politiepost', 'politieposten'],
    'brandtoren': ['brandtoren', 'brandtorens'],
    'pompstation': ['pompstation', 'pompstations', 'tankstation', 'tankstations'],
    'postkantoor': ['postkantoor', 'postkantoren', 'hulppostkantoor', 'postagentschap', 'postkantoor'],
    'radarpost': ['radarpost', 'radarposten'],
    'radartoren': ['radartoren', 'radartorens', 'radar'],
    'reddingboothuisje': ['reddingboothuisje', 'reddingboothuisje', 'reddingboothuis', 'reddingsboothuis'],
    'remise': ['remise', 'remises', 'remisen'],
    'brandweerkazerne': ['brandweerkazerne', 'brandweerkazernes', 'brandweer'],
    'schaapskooi': ['schaapskooi', 'schaapskooien'],
    'school': ['school', 'scholen'],
    'schoorsteen': ['schoorsteen', 'schoorstenen', 'schoorsteen', 'schoorsteenpijp.', 'schouw', 'schouwen'],
    'sporthal': ['sporthal', 'gymlokaal', 'gymnastieklokaal', 'gymzaal', 'sporthallen', 'sportschool', 'turnzaal'],
    'stadion': ['stadion', 'stadia', 'stadions'],
    'synagoge': ['synagoge', 'synagogen'],
    'tank': ['tank', 'brandstoftank', 'opslagtank', 'reservoir', 'ton', 'vat', 'tanks'],
    'bunker': ['bunker', 'bunker', 'bunkers'],
    'tankstation': ['tankstation', 'tankstations'],
    'telecommunicatietoren': ['telecommunicatietoren', 'telecommunicatietorens'],
    'toren': ['toren', 'torens'],
    'transformatorstation': ['transformatorstation', 'transformatorstationen'],
    'uitzichttoren': ['uitzichttoren', 'uitkijktorens', 'uitkijktoren', 'uitkijktoren'],
    'universiteit': ['universiteit', 'universiteiten', 'academie', 'hogeschool'],
    'veiling': ['veiling', 'veilingen', 'verkoping'],
    'vuurtoren': ['vuurtoren', 'vuurtorens'],
    'crematorium': ['crematorium', 'crematoria', 'crematoriums'],
    'waterradmolen': ['waterradmolen', 'waterradmolens'],
    'watertoren': ['watertoren', 'watertorens'],
    'werf': ['werf', 'werfs', 'werven'],
    'windmolen': ['windmolen', 'windmolens'],
    'windturbine': ['windturbine', 'windturbines', 'windturbinen'],
    'ziekenhuis': ['ziekenhuis', 'ziekenhuizen', 'hospitalen'],
    'dok': ['dok', 'doks', 'havenafdeling'],
    'zwembad': ['zwembad', 'zwembaden'],
    'elektriciteitscentrale': ['elektriciteitscentrale', 'energiecentrale', 'krachtcentrale'],
}

brt_properties_to_kad_con = {
    'fort': 'kad-con:fort',
    'gemaal': 'kad-con:gemaal',
    'gemeentehuis': 'kad-con:gemeentehuis',
    'gevangenis': 'kad-con:gevangenis',
    'huizenblok': 'kad-con:huizenblok',
    'kapel': 'kad-con:kapel',
    'kasteel': 'kad-con:kasteel',
    'kerk': 'kad-con:kerk',
    'koeltoren': 'kad-con:koeltoren',
    'kunstijsbaan': 'kad-con:kunstijsbaan',
    'lichttoren': 'kad-con:lichttoren',
    'luchtwachttoren': 'kad-con:luchtwachttoren',
    'manege': 'kad-con:manege',
    'moskee': 'kad-con:moskee',
    'observatorium': 'kad-con:observatorium',
    'paleis': 'kad-con:paleis',
    'parkeerdak, parkeerdek, parkeergarage': 'kad-con:parkeerdak_parkeerdek_parkeergarage',
    'politiebureau': 'kad-con:politiebureau',
    'brandtoren': 'kad-con:brandtoren',
    'pompstation': 'kad-con:pompstation',
    'postkantoor': 'kad-con:postkantoor',
    'radarpost': 'kad-con:radarpost',
    'radartoren': 'kad-con:radartoren',
    'reddingboothuisje': 'kad-con:reddingboothuisje',
    'remise': 'kad-con:remise',
    'brandweerkazerne': 'kad-con:brandweerkazerne',
    'schaapskooi': 'kad-con:schaapskooi',
    'school': 'kad-con:school',
    'schoorsteen': 'kad-con:schoorsteen',
    'sporthal': 'kad-con:sporthal',
    'stadion': 'kad-con:stadion',
    'synagoge': 'kad-con:synagoge',
    'tank': 'kad-con:tank',
    'bunker': 'kad-con:bunker',
    'tankstation': 'kad-con:tankstation',
    'telecommunicatietoren': 'kad-con:telecommunicatietoren',
    'toren': 'kad-con:toren',
    'transformatorstation': 'kad-con:transformatorstation',
    'uitzichttoren': 'kad-con:uitzichttoren',
    'universiteit': 'kad-con:universiteit',
    'veiling': 'kad-con:veiling',
    'vuurtoren': 'kad-con:vuurtoren',
    'crematorium': 'kad-con:crematorium',
    'waterradmolen': 'kad-con:waterradmolen',
    'watertoren': 'kad-con:watertoren',
    'werf': 'kad-con:werf',
    'windmolen': 'kad-con:windmolen',
    'windturbine': 'kad-con:windmolen',
    'ziekenhuis': 'kad-con:ziekenhuis',
    'dok': 'kad-con:dok',
    'zwembad': 'kad-con:zwembad',
    'elektriciteitscentrale': 'kad-con:elektriciteitscentrale',
}

NEWEST = [
    'nieuwste', 'laatste', 'meest recente', 'recentste'
]

OLDEST = [
    'oudste', 'eerste'
]

BIGGEST = [
    'grootste', 'grootte'
]

SMALLEST = [
    'kleinste', 'kleine', 'kleinste'
]

STREET = [
    'straat', 'laan'
]

CITY = [
    'woonplaats', 'stad', 'dorp', 'gehucht'
]

AMOUNT = [
    'aantal'
]

RETRIEVE = [
    'zien'
]

SCHOLEN = [
    'school', 'scholen', 'college', 'schoolgebouw', 'onderwijsinstituut', 'opleiding'
]

SUPERMARKTEN = [
    'supermarkt', 'supermarkten', 'winkel', 'winkels', 'shop'
]

OV_HALTES = [
    'ov haltes', 'ovhaltes', 'bushaltes', 'bushalten', 'station', 'stations', 'haltes'
]




'''
Note in the following piece of code we basically create TEMPLATE_OPTIONS which will be a set containing
formatted strings where we simply have '{order}', '{brtproperty}', ... etc


'''
class Template(Enum):
    ORDER = 'order'
    BRT_PROPERTY = 'brtproperty'
    YEAR = 'year'
    LOCATION = 'location'
    CBS_PROPERTY = 'cbsproperty'
    CBS_GRANULARITY = 'cbsgranularity'
    PROPERTY = 'property'
    GRANULARITY = 'granularity'
    TRANSPORTATION = 'transportation'
    AMOUNT = 'amount'
    PERMIT = 'permit'
    YEAR_FILTER = 'constructionyear'
    PARCEL_SIZE = 'parcelsize'
    HOUSE_SIZE = 'housesize'


def get_template_formatted(template):
    return f'{{{template}}}'


TEMPLATE_OPTIONS = {get_template_formatted(x.value) for x in Template}
