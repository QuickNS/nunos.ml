import requests
import time
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

class ComputerVisionService:
    
    _key = ''
    _base_url = ''

    def __init__(self, region, key):
        self._key = key
        self._base_url = '{}.api.cognitive.microsoft.com'.format(region)

    def analyzeUrl(self, imageUrl, visualFeatures='Categories, Tags, Description', details=None):
        
        params = {
            # Request parameters
            'visualFeatures': visualFeatures,
            }

        if details != None:
            params['details'] = details

        params = urllib.parse.urlencode(params)

        body = {"url": imageUrl}

        data = self._doPostRequest('/vision/v1.0/analyze', body, params)
        return data

    def analyzeImage(self, imageData, visualFeatures='Categories, Tags, Description', details=None):
               
        params = {
            # Request parameters
            'visualFeatures': visualFeatures,
            }

        if details != None:
            params['details'] = details

        params = urllib.parse.urlencode(params)

        data = self._sendFile('/vision/v1.0/analyze', imageData, params)
        return data

    def generateThumbnailForUrl(self, imageUrl, width, height, smartCropping=True):
        
        outputFile = 'thumb.jpg'
     
        params = urllib.parse.urlencode({
            # Request parameters
            'width': str(width),
            'height': str(height),
            'smartCropping': 'true' if smartCropping else 'false'
            })

        body = {"url": imageUrl}

        outputFile = self._getFile('/vision/v1.0/generateThumbnail', 'thumb.jpg', params, body)
        return outputFile

    def generateThumbnailForImage(self, imageData, width, height, smartCropping=True):
        
        outputFile = 'thumb.jpg'
     
        params = urllib.parse.urlencode({
            # Request parameters
            'width': str(width),
            'height': str(height),
            'smartCropping': 'true' if smartCropping else 'false'
            })

        outputFile = self._sendAndGetFile('/vision/v1.0/generateThumbnail', 'thumb.jpg', imageData, params)
        return outputFile

    def listDomainModels(self):

        data = self._doGetRequest('/vision/v1.0/models', None)
        return data

    def analyzeUrlWithModel(self, imageUrl, model):
        
        body = {"url": imageUrl}

        data = self._doPostRequest('/vision/v1.0/models/%s/analyze' % model, body, None)
        return data

    def analyzeImageWithModel(self, imageData, model):
        
        data = self._sendFile('/vision/v1.0/models/%s/analyze' % model, imageData, None)
        return data

    def describeUrl(self, imageUrl, maxCandidates=1):
        
        params = urllib.parse.urlencode({
            # Request parameters
            'maxCandidates' : maxCandidates,
            })

        body = {"url": imageUrl}

        data = self._doPostRequest('/vision/v1.0/describe', body, params)
        return data

    def describeImage(self, imageData, maxCandidates=1):
        
        params = urllib.parse.urlencode({
            # Request parameters
            'maxCandidates' : maxCandidates,
            })

        data = self._sendFile('/vision/v1.0/describe', imageData, params)
        return data

    def ocrWithUrl(self, imageUrl, language='unk', detectOrientation=True):
        
        params = urllib.parse.urlencode({
            # Request parameters
            'language' : language,
            'detectOrientation' : detectOrientation
            })

        body = {"url": imageUrl}

        data = self._doPostRequest('/vision/v1.0/ocr', body, params)
        return data

    def ocrWithImage(self, imageData, language='unk', detectOrientation=True):
        
        params = urllib.parse.urlencode({
            # Request parameters
            'language' : language,
            'detectOrientation' : detectOrientation
            })

        data = self._sendFile('/vision/v1.0/ocr', imageData, params)
        return data

    def recognizeTextWithUrl(self, imageUrl):
  
        url = '/vision/v1.0/RecognizeText'
        params = None

        body = {"url": imageUrl}
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/json'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, None), json.dumps(body), headers)
            response = conn.getresponse()
            conn.close()
            return response.headers['Operation-Location']
        except Exception as e:
            print('Error: %s' % str(e))
        

    def recognizeTextWithImage(self, imageData):
  
        url = '/vision/v1.0/RecognizeText'
        params = None
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/octet-stream'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), imageData, headers)
            response = conn.getresponse()
            conn.close()
            return response.headers['Operation-Location']
        except Exception as e:
            print('Error: %s' % str(e))
        

    def checkRecognizeTextStatus(self, url):

        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
        }

        analysis = self._doGetRequest(url, None)
        return analysis


    def _doGetRequest(self, url, params):
        
        headers = {
            # Request headers
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self._key,
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("GET", "%s?%s" % (url, params), None, headers)
            response = conn.getresponse()
            data = response.read()
            jData = json.loads(data.decode('utf8'))
            conn.close()
            return jData
        except Exception as e:
            print('Error: %s' % str(e))

    def _doPostRequest(self, url, body, params=None):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/json'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), json.dumps(body), headers)
            response = conn.getresponse()
            data = response.read()
            jData = json.loads(data.decode('utf8'))
            conn.close()
            return jData
        except Exception as e:
            print('Error: %s' % str(e))

    def _getFile(self, url, outputFile, params=None, body=None):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/json'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), json.dumps(body), headers)
            response = conn.getresponse()
            with open(outputFile, 'wb') as f:
                f.write(response.read())
            conn.close()
            return outputFile
        except Exception as e:
            print('Error: %s' % str(e))

    def _sendFile(self, url, imageData, params=None):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/octet-stream'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), imageData, headers)
            response = conn.getresponse()
            data = response.read()
            jData = json.loads(data.decode('utf8'))
            conn.close()
            return jData
        except Exception as e:
            print('Error: %s' % str(e))

    def _sendAndGetFile(self, url, outputFile, imageData, params=None):
        
        headers = {
            'Ocp-Apim-Subscription-Key': self._key,
            'Content-Type': 'application/octet-stream'
        }

        try:
            conn = http.client.HTTPSConnection(self._base_url)
            conn.request("POST", "%s?%s" % (url, params), imageData, headers)
            response = conn.getresponse()
            with open(outputFile, 'wb') as f:
                f.write(response.read())
            conn.close()
            return outputFile
           
        except Exception as e:
            print('Error: %s' % str(e))


  