#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from odf import text, teletype
from odf.opendocument import load
from tika import parser
import random
import time

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


# In[ ]:


def download_from_drive(file_id, destination):
    file_obj = drive.CreateFile({'id': file_id})
    file_obj.GetContentFile(destination)

        


# In[ ]:


def extraer_texto_odt(file_id):
    destination='doc/myfile_'+ file_id+ '.odt'
    download_from_drive(file_id, destination)
    textdoc = load(destination)
    allparas = textdoc.getElementsByType(text.P)
    texto=[]
    for i in allparas:
        texto.append(teletype.extractText(i))
    texto=' '.join(texto)
    return texto


# In[ ]:


def extraer_texto_pdf(file_id):
    from tika import parser
    destination = 'doc/myfile_'+ file_id+ '.pdf'
    download_from_drive(file_id, destination)
    file_data = parser.from_file(destination)
    return file_data['content'] 


# In[ ]:


def lista_texto(dataset):
    txt_declaracion=[]
    for item in dataset['LINK']:
        time.sleep(random.randint(2,12))
        if item: 
            id_file=extraer_id(item)
            try:
                titulo, extension= get_metadata(id_file)
                print(id_file, titulo, extension)   
                if extension=='application/pdf':
                    text=extraer_texto_pdf(id_file)
                else:
                    text=extraer_texto_odt(id_file)
            except:
                text="Falla"
                print(id_file, text)
            txt_declaracion.append(text)
        else:
            txt_declaracion.append('NADA')
    return txt_declaracion


def descargar_de_drive(dataset):
    txt_declaracion=[]
    for item in dataset:
        id_file=extraer_id(item)
        if id_file: 
            try:
                titulo, extension= get_metadata(id_file)
                print(id_file, titulo, extension)   
                if extension=='application/pdf':
                    text=extraer_texto_pdf(id_file)
                else:
                    text=extraer_texto_odt(id_file)
            except:
                text="Falla"
                print(id_file, text)
            txt_declaracion.append(text)
        else:
            txt_declaracion.append('NADA')
        time.sleep(random.randint(2,12))
    return txt_declaracion

def extraer_id(url):
    if "id=" in str(url):
        id_file= url.split('id=')[1]
    elif "/d/" in str(url):
        id_file=url.split('/')[5]
    else:
        id_file=None
    return id_file


def get_metadata(file_id):
    file = drive.CreateFile({'id': file_id})
    return file['title'], file['mimeType']
    
    
def cargar_desde_carpeta(dataset):
    txt_declaracion=[]
    n=0
    for item in dataset:
        n=n+1
        print(n)
        id_file=extraer_id(item)
        if id_file:   
            try:
                destination='doc/myfile_'+ id_file+ '.odt'
                textdoc = load(destination)
                allparas = textdoc.getElementsByType(text.P)
                texto=[]
                for i in allparas:
                    texto.append(teletype.extractText(i))
                texto=' '.join(texto)
            except:
                try:
                    from tika import parser
                    destination = 'doc/myfile_'+ id_file+ '.pdf'
                    file_data = parser.from_file(destination)
                    texto= file_data['content'] 
                except:
                    texto="Archivo no encontrado"
            txt_declaracion.append(texto)
        else:
            txt_declaracion.append('NADA')
    return txt_declaracion
    
 