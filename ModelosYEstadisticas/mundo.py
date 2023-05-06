from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination



#Las nuevas variables
var={
    "age":["Adulto","AdultoJoven","AdultoMayor"],
    "sex":['1.0','0.0'],
    "cp":['1.0','2.0','3.0','4.0'],
    #OJO HAY QUE ANIADIR HIPOTENSION AL TRESTBPS
    "trestbps":["TensionNormal","Hipertension",'Hipotension'],
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





#Deserializar modelo
readerP1=BIFReader('ModeloBic.bif')
readerPablo=BIFReader('modeloPablito.bif')
readerRestringido=BIFReader('ModeloRestringido.Bif')
readerBic=BIFReader('ModeloBic.bif')
readerK2=BIFReader('ModeloK2.bif')



#Traer el modelo
modeloP1=readerP1.get_model()
modeloPablo=readerPablo.get_model()
modeloRestringido=readerRestringido.get_model()
modeloBic=readerBic.get_model()
modeloK2=readerK2.get_model()



#Listas de las keys de cada variable
llavesP1=modeloP1.nodes()
llavesPablo=modeloPablo.nodes()
llavesRestringido=modeloRestringido.nodes()
llavesBic=modeloBic.nodes()
llavesK2=modeloK2.nodes()

#print({key: var[key] for key in llavesRestringido if key in var})

#Thresholds de los modelos
threshP1=0.18657636245651651
threshPablo=0.18657636245651651
threshRestringido=0.5
threshBic=0.4058099598551244
threshK2=0.42644593473350884



#Crear los infer
inferP1=VariableElimination(modeloP1)
inferPablo=VariableElimination(modeloPablo)
inferRestringido=VariableElimination(modeloRestringido)
inferBic=VariableElimination(modeloBic)
inferK2=VariableElimination(modeloK2)

#print(modeloP1.nodes())


#RECORDAR DE PONER TODAS LAS EVIDENCIAS COMO STRING
#print(inferP1.query(['hd'],evidence={'cp':'1.0'}).values[1])
#Funcion que retorna 1 si tiene hd, 0 dlc

def tengohd(nEvidencia):

    #Reviso la evidencia y despues hago los query:

    #P1
    p1Evidencia= {key: nEvidencia[key] for key in llavesP1 if key in nEvidencia}
    probP1=inferP1.query(['hd'],evidence=p1Evidencia).values[1]

    #Pablo
    pabloEvidencia={key: nEvidencia[key] for key in llavesPablo if key in nEvidencia}
    probPablo=inferPablo.query(['hd'],evidence=pabloEvidencia).values[1]

    #Restringido
    restringidoEvidencia={key: nEvidencia[key] for key in llavesRestringido if key in nEvidencia}
    probRestringido=inferRestringido.query(['hd'],evidence=restringidoEvidencia).values[1]

    #Bic
    bicEvidencia={key: nEvidencia[key] for key in llavesBic if key in nEvidencia}
    probBic=inferBic.query(['hd'],evidence=bicEvidencia).values[1]

    #K2
    k2Evidencia={key: nEvidencia[key] for key in llavesK2 if key in nEvidencia}
    probK2=inferP1.query(['hd'],evidence=k2Evidencia).values[1]

    #lo comparo con el threshold individual
    resP1=1 if probP1>threshP1 else 0 
    resPablo=1 if probP1>threshPablo else 0
    resRestringido=1 if probP1>threshRestringido else 0
    resBic=1 if probP1>threshBic else 0
    resK2=1 if probP1>threshK2 else 0

    #Hago la respuesta
    res=[(round(probP1,4),resP1),(round(probPablo,4),resPablo),
         (round(probRestringido,4),resRestringido),(round(probBic,4),resBic),
         (round(probK2,4),resK2)]

    return res

print(tengohd({'cp':'1.0','age':var['age'][0]}))