import http.client
import urllib.request
import urllib.parse
import urllib.error
import base64
import json
from xml.etree import ElementTree
import vlc

class TranslatorService:

    _key = ''
    _base_url = 'api.microsofttranslator.com'

    def __init__(self, key):
        self._key = key

    def translate(self, textToTranslate, to_lang, use_nn=True):

        category = 'generalnn' if use_nn else 'general'
        
        params = urllib.parse.urlencode({
        # Request parameters
        'text': textToTranslate,
        'to': to_lang,
        'category' : category
        })

        translation = self._doGetRequest('/V2/Http.svc/Translate', params)
        return translation.text

    def detect(self, textToDetect):
     
        params = urllib.parse.urlencode({
        # Request parameters
        'text': textToDetect,
        })

        detection = self._doGetRequest('/V2/Http.svc/Detect', params)
        return detection.text

    def getLanguagesForSpeak(self):
        
        response = self._doGetRequest('/V2/Http.svc/GetLanguagesForSpeak', None)
        language_list = []
        for child in response.iter('{http://schemas.microsoft.com/2003/10/Serialization/Arrays}string'):
            language_list.append(child.text)
        return language_list    

    def getLanguagesForTranslate(self):
        
        response = self._doGetRequest('/V2/Http.svc/GetLanguagesForTranslate', None)
        language_list = []
        for child in response.iter('{http://schemas.microsoft.com/2003/10/Serialization/Arrays}string'):
            language_list.append(child.text)
        return language_list    

    def getLanguageNames(self, langCodes, locale):
      
        params = urllib.parse.urlencode({
        # Request parameters
        'locale': locale,
        })
        
        strings_xml = ''
        for i in langCodes:
            strings_xml = strings_xml + '<string>{}</string>'.format(i)
        
        body = "<ArrayOfstring xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">" + \
                    strings_xml + \
                "</ArrayOfstring>"
    
    
        langs = self._doPostRequest('/V2/Http.svc/GetLanguageNames', body, params)
        language_list = []
        for child in langs.iter('{http://schemas.microsoft.com/2003/10/Serialization/Arrays}string'):
            language_list.append(child.text)
        return language_list    
            
    def translateArray(self, list_strings, to_lang, use_nn=True):
        
        category = 'generalnn' if use_nn else 'general'

        strings_xml = ''
        for i in list_strings:
            strings_xml = strings_xml + '<string xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">{}</string>'.format(i)
        
        body = "<TranslateArrayRequest>" + \
                    "<AppId />" + \
                    "<Options>" + \
                        "<Category xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\">{}</Category>".format(category) + \
                        "<ContentType xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\">text/plain</ContentType>" + \
                        "<ReservedFlags xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\" />" + \
                        "<State xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\" />" + \
                        "<Uri xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\" />" + \
                        "<User xmlns=\"http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2\" />" + \
                    "</Options>" + \
                    "<Texts>" + \
                    strings_xml + \
                    "</Texts>" + \
                    "<To>{}</To>".format(to_lang) + \
                    "</TranslateArrayRequest>"
              
        translations = self._doPostRequest('/V2/Http.svc/TranslateArray', body)
        translation_list = []
        for child in translations.iter('{http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2}TranslatedText'):
            translation_list.append(child.text)
        return translation_list 

    def detectArray(self, list_strings):
        
        strings_xml = ''
        for i in list_strings:
            strings_xml = strings_xml + '<string xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">{}</string>'.format(i)
        
        body = "<ArrayOfstring xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">" + \
                    strings_xml + \
                "</ArrayOfstring>"
        
        langs = self._doPostRequest('/V2/Http.svc/DetectArray', body)
        language_list = []
        for child in langs.iter('{http://schemas.microsoft.com/2003/10/Serialization/Arrays}string'):
            language_list.append(child.text)
        return language_list    

        
    def speak(self, text, language, options='MaxQuality|Female', format="audio/wav"):
        
        params = urllib.parse.urlencode({
        # Request parameters
        'text': text,
        'language': language,
        'options': options
        })

        outputFile = self._getFile('/V2/Http.svc/Speak', params, 'output.wav')

        p = vlc.MediaPlayer(outputFile)
        p.play()
        return outputFile


    def _doGetRequest(self, url, params):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("GET", "%s?%s" % (url, params), None, headers)
            response = conn.getresponse()
            data = response.read()
            xml = ElementTree.fromstring(data.decode('utf-8'))
            conn.close()
            return xml
        except Exception as e:
            print('Error: %s' % str(e))

    def _doPostRequest(self, url, body, params=None):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type' : 'text/xml; charset=utf-8'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), body.encode('utf8'), headers)
            response = conn.getresponse()
            data = response.read()
            xml = ElementTree.fromstring(data.decode('utf-8'))
            conn.close()
            return xml
        except Exception as e:
            print('Error: %s' % str(e))

    def _getFile(self, url, params, outputFile):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("GET", "%s?%s" % (url, params), None, headers)
            response = conn.getresponse()
            with open(outputFile, 'wb') as f:
                f.write(response.read())
            conn.close()
            return outputFile
        except Exception as e:
            print('Error: %s' % str(e))

