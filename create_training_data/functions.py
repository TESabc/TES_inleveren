import re
import logging
import requests
import urllib.parse
from shapely import from_wkt # this needs to be installed manually!!!
# ( i did it in pycharm, think about requirements file)
from create_training_data.query_components import SpecialTokens, ADDRESS_QUESTION

SUGGEST_URL = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
              '?fq=type:(gemeente OR woonplaats OR adres OR provincie) AND bron:BAG&q={}'


class AddressFunction:

    custom_arguments = {
        'gemeentecode',
        'provinciecode',
        'straatnaam_url',
        'huis_nlt_url',
        'postcode_url',
        'woonplaatsnaam_url',
        'centroide_rd_url',
        'centroide_rd_x',
        'centroide_rd_y',
        'address',
    }

    @staticmethod
    def get_argument(argument, data, address_str, query):
        if argument == 'gemeentecode':
            result = f"GM{data[argument]}"
        elif argument == 'provinciecode':
            result = f'00{data[argument][-2:]}'
        elif argument == 'address':
            result = address_str
        elif argument == 'centroide_rd_url':
            if f'#{argument}' in query:
                result = ','.join([str(coordinate) for coordinate in from_wkt(data['centroide_rd']).coords[0]])
            else:
                result = None
        elif argument == 'centroide_rd_x':
            if f'#{argument}' in query:
                result = from_wkt(data['centroide_rd']).coords[0][0]
            else:
                result = None
        elif argument == 'centroide_rd_y':
            if f'#{argument}' in query:
                result = from_wkt(data['centroide_rd']).coords[0][1]
            else:
                result = None
        elif argument.endswith('_url'):
            formatted_argument = argument[:-4]
            if f'#{formatted_argument}' in query:
                result = urllib.parse.quote_plus(str(data[formatted_argument]))
            else:
                result = None
        else:
            result = data.get(argument)

        return result


class Functions:

    @staticmethod
    def generate_function_text(function, *args):
        joined_args = f' {SpecialTokens.SEPARATOR.value} '.join(str(arg) for arg in args)
        return f'{function.value} {joined_args}'

    @staticmethod
    def format_result(kwargs, address_str, query):
        custom_kwargs = {}
        for k in AddressFunction.custom_arguments:
            custom_kwargs[k] = AddressFunction.get_argument(k, kwargs, address_str, query)

        all_kwargs = {**kwargs, **custom_kwargs}
        result = {
            k: all_kwargs[k] for k in sorted(all_kwargs.keys(), key=len, reverse=True)
        }

        return result

    @staticmethod
    def format_url_result(kwargs):
        result = {}
        for k, v in kwargs.items():
            if k == 'centroide_rd':
                v = ','.join([str(coordinate) for coordinate in from_wkt(v).coords[0]])
            else:
                v = urllib.parse.quote_plus(str(v))

            result[k] = v

        rd_centroid = from_wkt(kwargs['centroide_rd']).coords[0]
        result.update({
            'centroide_rd_x': rd_centroid[0],
            'centroide_rd_y': rd_centroid[1],
        })

        result = {
            k: result[k] for k in sorted(result.keys(), key=len, reverse=True)
        }

        return result

    @staticmethod
    def get_address(address):
        url = SUGGEST_URL.format(address)
        address = requests.get(url).json()['response']['docs'][0]

        return address

    @staticmethod
    def apply_functions(prediction):
        if prediction.startswith(SpecialTokens.LOOKUP_ADDRESS.value):
            function = SpecialTokens.LOOKUP_ADDRESS
            query = prediction[len(function.value):]
        else:
            function = None
            query = prediction
        # We will return the preflabel or code of the location to include instance data description of in-context examples
        location_code_or_preflabel = None
        label_corresponding_to_code = None

        address_data = None
        if function == SpecialTokens.LOOKUP_ADDRESS:
            query = query.strip()

            components = query.partition(SpecialTokens.SEPARATOR.value)
            components = [component.strip() for component in components if component != SpecialTokens.SEPARATOR.value]
            if len(components) > 1:
                address_str, query = components
                try:
                    address_data = Functions.get_address(address_str)
                    address_data_processed = Functions.format_result(address_data, address_str, query)

                    for key, value in address_data_processed.items():
                        query_new = re.sub(f"#{key}", str(value), query)
                        # If a substitution is made we record the preflabel or code
                        if query_new != query:
                            location_code_or_preflabel = str(value)
                            if key == 'gemeentecode':
                                label_corresponding_to_code = address_data_processed['gemeentenaam']
                            elif key == 'provinciecode':
                                label_corresponding_to_code = address_data_processed['provincienaam']
                        else:
                            pass
                        query = query_new
                except Exception as e:
                    logging.exception(e)
                    query = ADDRESS_QUESTION

        return query, location_code_or_preflabel, label_corresponding_to_code
