import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash import Dash, dash_table
import pandas as pd
import base64
import os
import dash_bootstrap_components as dbc
import json
from pgmpy.models import BayesianNetwork;from pgmpy.factors.discrete import TabularCPD;import dash;from dash import dcc ;from dash import html;from dash.dependencies import Input, Output;import plotly.express as px;from pgmpy . inference import VariableElimination
from pgmpy . sampling import BayesianModelSampling;from pgmpy . estimators import MaximumLikelihoodEstimator;
from pgmpy . estimators import BayesianEstimator;import pandas as pd;from pgmpy . inference import VariableElimination
import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination

#LOS DATOS LOS TIENE QUE BAJAR DE AWS Y LLAMARLO DATOS

#Datos
#import dash_bootstrap_components as dbc
external_stylesheets = [dbc.themes.BOOTSTRAP]
#datos = pd.read_csv('processed.cleveland.data', header=None, delimiter=',')
datos = pd.read_csv('processed.cleveland.data', header=None, delimiter=',')

nameDatos=['age','sex','cp','trestbps','chol',
           'fbs','restecg','thalach','exang',
           'oldpeak','slope','ca','thal','hd']
datos.columns=nameDatos
#print(datos.head())



#Limpieza de datos
#Quitar los ?
filasBorrar=datos[datos.eq('?').any(axis=1)]
datos=datos.drop(filasBorrar.index)

#Thalach

datos['MaxHB']=220-datos['age']
datos['thalach2.0']=datos['thalach']
datos.loc[datos['thalach2.0']<=0.77*datos['MaxHB'],'thalach']='Bajo'
datos.loc[(datos['thalach2.0']>0.77*datos['MaxHB']) & (datos['thalach2.0']<=0.93*datos['MaxHB']),'thalach']='Normal'
datos.loc[datos['thalach2.0']>0.93*datos['MaxHB'],'thalach']='Alta'
datos.drop(['MaxHB','thalach2.0'],axis=1,inplace=True)
#print(datos.head())

#age

def transformEdad(age):
    if age<=39:
        return "AdultoJoven"
    elif age>=40 and age<=59:
        return "Adulto"
    else:
        return "AdultoMayor" 
datos['age']=datos['age'].apply(transformEdad)

#presion

def transformPresion(presion):
    if presion <95:
        return "Hipotension"
    elif presion>=95 and presion<=140:
        return "TensionNormal"
    else:
        return "Hipertension"
datos['trestbps']=datos['trestbps'].apply(transformPresion)

#Cholesterol

def transformChol(chol):
    if chol<193:
        return "Normal"
    else:
        return "Alto"   
datos['chol']=datos['chol'].apply(transformChol)

#hd

def transformhd(hd):
    if hd==0:
        return "No hay hd"
    else:
        return "Si hay hd" 
datos['hd']=datos['hd'].apply(transformhd)
def transformoldpeak(oldpeak):
    if oldpeak<=1:
        return "Normal"
    else:
        return "Alto"
datos['oldpeak']=datos['oldpeak'].apply(transformoldpeak)
#print(datos.head())
def return_dict_as_string(dictionary):
    return json.dumps(dictionary)

def retornarEvidencia():
    evidencia={}
    llaves=list(var.keys())
    if vV1!="":
        evidencia[llaves[1]]=var[llaves[1]][vV1]
    if vV2!="":
        evidencia[llaves[2]]=var[llaves[2]][vV2]
    if vV3!="":
        evidencia[llaves[3]]=var[llaves[3]][vV3]
    if vV4!="":
        evidencia[llaves[4]]=var[llaves[4]][vV4]
    if vV5!="":
        evidencia[llaves[5]]=var[llaves[5]][vV5]
    if vV6!="":
        evidencia[llaves[6]]=var[llaves[6]][vV6]
    if vV7!="":
        evidencia[llaves[7]]=var[llaves[7]][vV7]
    if vV8!="":
        evidencia[llaves[8]]=var[llaves[8]][vV8]
    if vV9!="":
        evidencia[llaves[9]]=var[llaves[9]][vV9]
    if vV10!="":
        evidencia[llaves[10]]=var[llaves[10]][vV10]
    if vV11!="":
        evidencia[llaves[11]]=var[llaves[11]][vV11]
    if vV12!="":
        evidencia[llaves[12]]=var[llaves[12]][vV12]
    if vV13!="":
        evidencia[llaves[13]]=var[llaves[13]][vV13]
    return evidencia

df2= datos.value_counts(["age","sex","hd"]).reset_index().rename(columns={0:"Conteo"})
fig = px.bar(df2, y="Conteo", x= "age", color="hd", barmode="group",facet_row="sex", text_auto=True)#

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

global vV1
vV1=""
vV2=""
vV3=""
vV4=""
vV5=""
vV6=""
vV7=""
vV8=""
vV9=""
vV10=""
vV11=""
vV12=""
vV13=""



var={
    "age":["Adulto","AdultoJoven","AdultoMayor"],
    "sex":['1.0','0.0'],
    "cp":['1.0','2.0','3.0','4.0'],
    #OJO HAY QUE ANIADIR HIPOTENSION AL TRESTBPS
    "trestbps":["TensionNormal","Hipertension"],
    "chol":["Alto","Normal"],
    "fbs":['1.0','0.0'],
    "restecg":['0.0','1.0','2.0'],
    "thalach":["Bajo","Normal","Alta"], # updated
    "exang":['1.0','0.0'],
    "oldpeak":["Normal","Alto"],
    "slope":['1.0','2.0','3.0'],
    "ca":['0.0','1.0','2.0','3.0'],
    "thal":['3.0','6.0','7.0']
}
varUsuario={
    "Edad":["Adulto","AdultoJoven","AdultoMayor"],
    "sexo":["Masculino","Femenino"],
    "Tipo de dolor de pecho":["Angina tipica","Angina atipica","Dolor no angina","Asintomatico"],
    "Presion arterial en reposo":["TensionNormal","Hipertension"],
    "colesterol":["Colesterol Alto","Colesterol Normal"],
    "Azucar en la sangre":["Mayor a 120mg/dl","Menor a 120mg/dl"],
    "Resultados electrogardiografo":["Normal","Abnormalidad ST-T","hipertrofia ventricular izquierda"],
    "Max ritmo cardiaco":["Bajo","Normal","Alta"], # updated
    "Angina producida por ejercicio":["Si","No"],
    "Depresion ST relativo al reposo":["Normal","Alto"],
    "Segmento ST peak":["Pendiente arriba","Plano","PendienteAbajo"],
    "Numero de mayores vasos sanguineos":['0.0','1.0','2.0','3.0'],
    "thalasemia":['Normal','defecto fijo','defecto reversible']
}
respuesta = ""
#OPCIONES DE SELECCIÓN
optAge = [
    {'label': 'Adulto', 'value': '0'},
    {'label': 'AdultoJoven', 'value': '1'},
    {'label': 'AdultoMayor', 'value': '2'}
]
optSex = [
    {'label': 'Masculino', 'value': '0'},
    {'label': 'Femenino', 'value': '1'}
]
optCp= [
    {'label': 'Angina Tipica', 'value': '0'},
    {'label': 'Angina Atipica', 'value': '1'},
    {'label': 'Sin dolor Angina', 'value': '2'},
    {'label': 'Asintomatico', 'value': '3'}
]
opttrestbps = [
    {'label': 'TensionNormal', 'value': '0'},
    {'label': 'Hipertension', 'value': '1'}
]
optchol = [
    {'label': 'Alto', 'value': '0'},
    {'label': 'Normal', 'value': '1'}
]
optfbs = [
    {'label': 'Mayor a 120mg/dl', 'value': '0'},
    {'label': 'Menor a 120mg/dl', 'value': '1'}
]
optrestecg = [
    {'label': 'Normal', 'value': '0'},
    {'label': 'Abnormalidad ST-T', 'value': '1'},
    {'label': 'hipertrofia ventricular izquierda', 'value': '2'}
]
optthalach = [
    {'label': 'Bajo', 'value': '0'},
    {'label': 'Normal', 'value': '1'},
    {'label': 'Alta', 'value': '2'}
]
optexang = [
    {'label': 'Si', 'value': '0'},
    {'label': 'No', 'value': '1'}
]
optoldpeak = [
    {'label': 'Normal', 'value': '0'},
    {'label': 'Alto', 'value': '1'}
]
optslope = [
    {'label': 'Pendiente arriba', 'value': '0'},
    {'label': 'Plano', 'value': '1'},
    {'label': 'PendienteAbajo', 'value': '2'}
]
optca = [
    {'label': '0.0', 'value': '0'},
    {'label': '1.0', 'value': '1'},
    {'label': '2.0', 'value': '2'},
    {'label': '3.0', 'value': '3'}
]
optthal = [
    {'label': 'Normal', 'value': '0'},
    {'label': 'defecto fijo', 'value': '1'},
    {'label': 'defecto reversible', 'value': '2'}
]


optVar = [
    {'label': 'age', 'value': 'age'},
    {'label': 'sex', 'value': 'sex'},
    {'label': 'cp', 'value': 'cp'},
    {'label': 'Trestbps', 'value': 'trestbps'},
    {'label': 'chol', 'value': 'chol'},
    {'label': 'fbs', 'value': 'fbs'},
    {'label': 'restecg', 'value': 'restecg'},
    {'label': 'thalach', 'value': 'thalach'},
    {'label': 'exang', 'value': 'exang'},
    {'label': 'oldpeak', 'value': 'oldpeak'},
    {'label': 'slope', 'value': 'slope'},
    {'label': 'ca', 'value': 'ca'},
    {'label': 'Thal', 'value': 'thal'}
]



# HTML

image_filename = "C:/Users/Santiago Zubieta/Documents/2023-1/AC/ProyectoACTD/ModelosSerializados/SenecaDoc.png"



def verificar_imagen(nombre_imagen):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(directorio_actual, nombre_imagen)
    
    if os.path.exists(ruta_imagen):
        return True
    else:
        return False
esta_en_directorio = verificar_imagen(image_filename)
print(esta_en_directorio)

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

app.layout = html.Div([     

    html.H1(children="SenecaDoc ",style={"fontSize": "64px", "color": "#569AAD","font-family": 'Oxygen',},),
         
    html.H2(children="Por favor ingresa la información a continuación para que SenecaDoc te ayude a realizar predicciones del riesgo de sufrir una enfermedad cardíaca",
        style={"fontSize": "20px", "color": "black","font-family": 'Oxygen',},),
    
     dbc.Row([
        dbc.Col(
            html.Div(["¿Cuál es el rango de edad?",
            dcc.Dropdown(id='dropdownAge',options=optAge),]),width=4), #,persistence=True, persistence_type='session'
        dbc.Col(
            html.Div(["¿Cuál es el sexo biológico?",
            dcc.Dropdown(id='dropdownSex',options=optSex,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Qué tipo de dolor de pecho?",
            dcc.Dropdown(id='dropdownCPT',options=optCp,value='',),]),width=4),                       
        dbc.Col(
            html.Div(["¿Cuál representa la presion arterial en reposo?",
            dcc.Dropdown(id='dropdowntrestbps',options=opttrestbps,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál es el nivel de colesterol?",
            dcc.Dropdown(id='dropdownchol',options=optchol,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál es el nivel de Azucar en la sangre?",
            dcc.Dropdown(id='dropdownfbs',options=optfbs,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál fue el resultados del electrogardiografo?",
            dcc.Dropdown(id='dropdownrestecg',options=optrestecg,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál ha sido su Maximo ritmo cardiaco?",
            dcc.Dropdown(id='dropdownthalach',options=optthalach,value='',),]),width=4),
         dbc.Col(
            html.Div(["¿Presenta Angina producida por ejercicio?",
            dcc.Dropdown(id='dropdownexang',options=optexang,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál es la depresion ST relativo al reposo?",
            dcc.Dropdown(id='dropdownoldpeak',options=optoldpeak,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál es el Segmento ST peak?",
            dcc.Dropdown(id='dropdownslope',options=optslope,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuánto es el número de mayores vasos sanguineos?",
            dcc.Dropdown(id='dropdownca',options=optca,value='',),]),width=4),
        dbc.Col(
            html.Div(["¿Cuál esel  valor de thalasemia?",
            dcc.Dropdown(id='dropdownthal',options=optthal,value='',),]),width=4),
        dbc.Col(
            html.Div(id='OutRespuesta',),width=4),
        
        
        html.Button(id='submit-button-state', n_clicks=0, children='Preguntarle probabilidad de enfermedad cardiaca a SenecaDoc'),
        html.Div(id='output-state',style={"fontSize": "32px", "color": "#569AAD"},),

        html.A(html.Button('Limpiar el formulario'),href='/'),
        ],align="center"),#,]),,align="center"    
     
        html.H2(
        children="Recomendaciones de SenecaDoc: ",
        style={"fontSize": "32px", "color": "#569AAD","font-family": 'Oxygen',},  
        ),
        html.H2(
        children="¿Quieres saber cuál es el mejor exámen que puedes hacer a continuación?",
        style={"fontSize": "24px", "color": "black","font-family": 'Oxygen',},  
        ),

        dbc.Row([
            dbc.Col(
                html.Div(id='OutRespuesta2',style={"fontSize": "24px", "vertical-align":"middle","margin":"50px 50px", "color": "#569AAD","font-family": 'Oxygen',},),),
            dbc.Col(    
                html.Img(src=b64_image(image_filename),width="200", height="250",style={"vertical-align":"middle","margin":"50px 400px"},),),
        ],align="center"),

        

        html.H2(children="¿Quieres ver un resumen gráfico algunas varibales utilizadas de tu interes? ",
        style={"fontSize": "32px", "color": "#569AAD","font-family": 'Oxygen',},  
        ),

        #GRAFICA Y SUS BOTONES
        dbc.Row([
            dbc.Col(
                html.Div(["Selecciona la variable para el eje X",
                    #style={"fontSize": "16px", "color": "black",},
                    dcc.Dropdown(
                        id='dropdownV1',
                        options=optVar,
                        value='age',
                    # style= {"width" : "400px",},
                    ),
                    html.Div(id='outputV1'),
            ]),width=4),
            dbc.Col([
                        html.Div("Selecciona la variable para el eje secundario Y",
                        #style={"fontSize": "16px", "color": "black",},
                        ),
                        dcc.Dropdown(
                            id='dropdownV2',
                            options=optVar,
                            value='sex',
                            #style= {"width" : "400px",},
                        ),
                        html.Div(id='outputV2')
                    ],width=4),
            

                ],align="center",),
                

     dcc.Graph(id='figura1',figure=fig),
 html.Div(id='hd1', style={'display':'none'}),
 html.Div(id='hd2', style={'display':'none'}),  
 html.Div(id='hd3', style={'display':'none'}),  
 html.Div(id='hd4', style={'display':'none'}),
 html.Div(id='hd5', style={'display':'none'}),
 html.Div(id='hd6', style={'display':'none'}),
 html.Div(id='hd7', style={'display':'none'}),
 html.Div(id='hd8', style={'display':'none'}),
 html.Div(id='hd9', style={'display':'none'}),
 html.Div(id='hd10', style={'display':'none'}),
 html.Div(id='hd11', style={'display':'none'}),
 html.Div(id='hd12', style={'display':'none'}),
 html.Div(id='hd13', style={'display':'none'}),    
])




#Acá va el modelo
readerP1=BIFReader('C:/Users/Santiago Zubieta/Documents/2023-1/AC/ProyectoACTD/ModelosSerializados/ModeloP1SZ.bif')
modelo=readerP1.get_model()
threshP1=0.18657636245651651

#Ajuste parametros

def printCPds():
    for i in modelo.nodes():
        print(modelo.get_cpds(i))

infer=VariableElimination(modelo)
limites=[0.1,0.5,0.8]
#Funcion hd:
    #Toma como parámetro la evidencia dada y retorna un diccionario que contiene en la primera llave la probabilidad de tener hd y en la segunda el veredicto
def tengohd(evidencia):
    try:
        busqueda=infer.query(["hd"],evidence=evidencia)
        p1=busqueda.values[0]
        p2=busqueda.values[1]
        respuesta=None
        respuesta=1 if p2>threshP1 else 0
        return {"Si hay hd": p2,"Veredicto":respuesta}
    except ValueError:
        return 'Combinacion de variables no validas'
print(tengohd({"age":"AdultoJoven", "oldpeak":"Alto","sex":"1.0"}))

#FUNCION QUE DEVUELVE LAS VARIABLES NO USADAS
def retornarNoEntrados(evidencia):
    llaves = list(evidencia.keys())
    noUsoVar = {}
    for clave in var.keys():
        if clave not in llaves:
            noUsoVar[clave] = var[clave]
    return noUsoVar


#fUNCION DE MAX DE CERTEZA:
#Función que retorna la variable no usada que maximiza la certeza de tener una enfermedad al corazón
#Entra por parámetro la evidencia usada
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def maxCerteza(evidencia):
    exNoUsados=retornarNoEntrados(evidencia)
    certezaActual=tengohd(evidencia)["Si hay hd"]
    variables=list(exNoUsados.keys())
    maximo=certezaActual
    lista={}
    for llaves in variables:
        turrones=[]
        for valores in var[llaves]:
            evidencia[llaves]=valores
            certezaNueva= tengohd(evidencia)["Si hay hd"]
            turrones.append(certezaNueva)
            del evidencia[llaves]
        promedio=sum(turrones)/len(turrones)
        if promedio>maximo:
            lista={}
            lista[llaves]=promedio
            maximo=promedio
    if maximo==certezaActual:
        lista['Respuesta']="No hay examenes de que aumenten la certeza actual"
    return lista
def retornarNombre(variable):
    nombre1=list(variable.keys())
    nombre2=nombre1[0]
    if nombre2=='Respuesta':
        return 'El valor esperado de añadir un examen es menor que la certeza actual'
    else:
        llaves=list(var.keys())
        llaves2=list(varUsuario.keys())
        pos=llaves.index(nombre2)
        #pos=llaves2.index(nombre2)
        return "Se recomienda hacer el siguiente examen: " + llaves2[pos]
    
#Call back del modelo
# Revisa parametros y calcula la probabilidad, luego la muestra    
@app.callback(Output('OutRespuesta', 'children'),
              Output('OutRespuesta2', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('dropdownAge', 'value'),
              State('dropdownSex', 'value'),
              State('dropdownCPT', 'value'),
              State('dropdowntrestbps', 'value'),
              State('dropdownchol', 'value'),
              State('dropdownfbs', 'value'),
              State('dropdownrestecg', 'value'),
              State('dropdownthalach', 'value'),
              State('dropdownexang', 'value'),
              State('dropdownoldpeak', 'value'),
              State('dropdownslope', 'value'),
              State('dropdownca', 'value'),
              State('dropdownthal', 'value'))
def update_output( n_clicks, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13):
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    laRespuesta = ""
    evidencia={}
    llaves=list(var.keys())
    if input1!= None and input1 !="" and trigger == "submit-button-state":
        evidencia[llaves[0]]=var[llaves[0]][int(input1)]
    if input2!= None and input2!= "" and trigger == "submit-button-state":
        evidencia[llaves[1]]=var[llaves[1]][int(input2)]
    if input3!= None and input3!="" and trigger=="submit-button-state":
        evidencia[llaves[2]]=var[llaves[2]][int(input3)]
    if input4!= None and input4!="" and trigger=="submit-button-state":
        evidencia[llaves[3]]=var[llaves[3]][int(input4)]
    if input5!= None and input5!="" and trigger=="submit-button-state":
        evidencia[llaves[4]]=var[llaves[4]][int(input5)]
    if input6!= None and input6!="" and trigger=="submit-button-state":
        evidencia[llaves[5]]=var[llaves[5]][int(input6)]
    if input7!= None and input7!="" and trigger=="submit-button-state":
        evidencia[llaves[6]]=var[llaves[6]][int(input7)]
    if input8!= None and input8!="" and trigger=="submit-button-state":
        evidencia[llaves[7]]=var[llaves[7]][int(input8)]
    if input9!= None and input9!="" and trigger=="submit-button-state":
        evidencia[llaves[8]]=var[llaves[8]][int(input9)]
    if input10!= None and input10!="" and trigger=="submit-button-state":
        evidencia[llaves[9]]=var[llaves[9]][int(input10)]
    if input11!= None and input11!="" and trigger=="submit-button-state":
        evidencia[llaves[10]]=var[llaves[10]][int(input11)]
    if input12!= None and input12!="" and trigger=="submit-button-state":
        evidencia[llaves[11]]=var[llaves[11]][int(input12)]
    if input13!= None and input13!="" and trigger=="submit-button-state":
        evidencia[llaves[12]]=var[llaves[12]][int(input13)]

    SiguienteExamen=""
    if evidencia!={}:
        laRespuesta="La probabilidad de que se tenga una enfermedad cardiaca es de: " +str(round(tengohd(evidencia)['Si hay hd'],4))
        res1=maxCerteza(evidencia)
        SiguienteExamen=retornarNombre(res1)
 
    return laRespuesta,SiguienteExamen


#GRAFICA Y SUS BOTONES

vV100="age"
vV200="sex"

@app.callback(
    Output('figura1',component_property='figure'),
    [dash.dependencies.Input('dropdownV1', 'value')],   
    [dash.dependencies.Input('dropdownV2', 'value')])
def update_figure(value,value2):
    global vV1, vV2, fig
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if trigger == "dropdownV1":
        if value == vV2:
            vV2=vV1
            vV1=value
        else:
           vV1=value 
        if vV2 == "":
            vV2="sex"
        df2= datos.value_counts([value,vV2,"hd"]).reset_index().rename(columns={0:"Conteo"})
        fig = px.bar(df2, y="Conteo", x= value, color="hd", barmode="group",facet_row=vV2)   #, facet_col="Resting-blood-pressure" 
    elif trigger == "dropdownV2":
        if value == vV2:
            vV1=vV2
            vV2=value2
        else:
           vV2=value2
        
        if vV1 == "":
            vV1="age"
        df2= datos.value_counts([vV1,value2,"hd"]).reset_index().rename(columns={0:"Conteo"})
        fig = px.bar(df2, y="Conteo", x= vV1, color="hd", barmode="group",facet_row=value2)   #, facet_col="Resting-blood-pressure" 


    return fig



if __name__ == '__main__':
    app.run_server(debug=True)