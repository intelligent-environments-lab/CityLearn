import random
import time
import datetime
import pandas as pd 
import numpy as np
from scipy.stats import skewnorm, truncnorm



# Definindo as hipóteses e as suas probabilidades
random.seed(time.time_ns())


CASAS_TIME = [0.25, 0.50, 0.75, 0.00] #TODO
TRUNK_MULTIPLE = 0.25 #TODO
ROTINA_CHANGE = 0.1 # Hipotese de ter uma trip time diferente na semana
MAX_BATTERY_CAPACITY = 70
CHARGE_AT_WORK = 0.06
CHARGER_CHANGE = 0.05
YEARS = 1

charger_ex = [
    ["11",["charger_13_1"],60,16],
    ["12",["charger_14_1"],80,18],
#    ["3",["charger_10_1"],75,18],
#    ["4",["charger_12_1"],45,15],
#    ["5",["charger_15_1"],90,20],
#    ["6",["charger_15_2"],100,21]
]
charger_usable = [
]

semana_dia = [ # Prob ; Min Left ; Max Left 
    ("Normal1", 0.70,7,9), # 7-8  
    ("Cedo", 0.25,5,6), # 5-6  
    ("tarde", 0.05,10,12), # 9-12   
]
semana_noite = [ # Prob ; Min Arrive ; Max Arrive 
    ("Normal1", 0.70,17,19), # 17-18
    ("tarde", 0.23,20,22), # 20-22  
    ("cedo", 0.07,14,16), # 14-16
]

fimdesemana = [ # Prob ; Min Left ; Max Left ; Min Arrive ; Max Arrive 
    ("casa", 0.20,-1,-1,-1,-1), # dia em case 
    ("manha", 0.45,8,11,10,15), # compras de manha e tarefas  8-11 / 10-15
    ("tarde", 0.30,11,14,14,23), # sair de tarde 11-14 / 14--22
    #("noite", 0.05,19,23,22,4), # sair á noite 19-23 / 22-4
]

dist_home = [ # Prob ; Min dist ; Max dist ; Min vel ; Max vel
    ("close", 0.35,3,10,20,30), 
    ("medium", 0.4,10,30,40,50), 
    ("far", 0.2,40,100,60,80),
]

dist_home_weekend = [ # Prob ; Min dist ; Max dist ; Min vel ; Max vel
    ("close", 0.20,3,10,20,30), 
    ("medium", 0.35,10,30,40,50), 
    ("far", 0.3,40,100,60,80),
    ("veryfar", 0.15,100,200,70,100),
]

traffic_week = [ # Prob ; Min time ; Max time 
    ("sem", 0.05,0,0),
    ("leve", 0.35,0.1,0.25), 
    ("medio", 0.4,0.3,0.5), 
    ("pesado", 0.2,0.6,1),   
]

traffic_weekend = [ # Prob ; Min time ; Max time 
    ("sem", 0.1,0,0),
    ("leve", 0.55,0.1,0.25), 
    ("medio", 0.25,0.3,0.5), 
    ("pesado", 0.1,0.6,1),   
]

energy = { 
    "spend":[15,30],
}





def get_random_cena(arr): # Função usada para gerar as cenas aleatórias com o index do mesmo
    # Seed variando com o tempo
    random.seed(time.time_ns())
    cena_aleatoria = random.choices(
        population = [h[0] for h in arr], 
        weights = [h[1] for h in arr], k=1)
    return {"name":cena_aleatoria[0],"index":[h[0] for h in arr].index(cena_aleatoria[0])}


# Test Print Cena
#print("Cena aleatória dia: ", get_random_cena(semana_dia)["name"])
#print("Cena aleatória noite: ", get_random_cena(semana_noite)["name"])
#print("Cena aleatória fim de semana: ", get_random_cena(fimdesemana)["name"])

# Random number BAKUP #TODO
'''def gerar_numero_aleatorio_especifico(minimo, maximo,control):
    random.seed(time.time_ns())
    casas_decimais = CASAS_TIME
    random_decimal = random.choice(casas_decimais)

    if(minimo > maximo):
        result = 24 - minimo
        result = result + maximo
        random_number = random.randint(0, int(result)) 
        trunk = (minimo + random_number)
        if(24 <= trunk):
            random_number = trunk-24
        
    else:
        random_number = random.randint(int(minimo), int(maximo))
 
    return random_number + random_decimal'''

def gerar_numero_aleatorio_especifico(min_value, max_value, control,skew):
    size=1
    std = 0.5
    mean = (min_value + max_value) / 2
    # Convertendo a medida da assimetria para o parâmetro da distribuição
    a = skew / np.sqrt(1 + skew**2)

    # Gerando valores da distribuição
    values = skewnorm.rvs(a, loc=mean, scale=std, size=size)

    # Aplicando os limites
    clipped_values = np.clip(values, min_value, max_value)
    
    return int(clipped_values[0])
# Random Time Day Night
def gen_time(cena,choice,min,max,control,skew):

    random.seed(time.time_ns())
    #print("\n\n\n\n")
    #print(choice["name"])
    cena_especifica = cena[choice["index"]]
    
    min_value = cena_especifica[min]
    if(min_value==-1):
        return -1
    if(control > min_value and not control == -1): 
        min_value=control
    #print(min_value)
    max_value = cena_especifica[max]
    #print(max_value)
    
    number = gerar_numero_aleatorio_especifico(min_value,max_value,control,skew)
    while(control >= number and number >= 5):
        number = gerar_numero_aleatorio_especifico(min_value,max_value,control,skew)
    
    return number

# Get simulation of departure and arrival of home

    

def gen_all(random_cena_dia,random_cena_noite,choice):
    random.seed(time.time_ns())
    normal_day = gen_time(semana_dia,random_cena_dia,2,3,-1,-10)
    normal_night = gen_time(semana_noite,random_cena_noite,2,3,normal_day,10)

    
    weekend_day = gen_time(fimdesemana,choice,2,3,-1,0)
    
    if(weekend_day==-1):
        weekend_night=-1
    else:
        weekend_night = gen_time(fimdesemana,choice,4,5,weekend_day,0)
    #print("\n")
    #print("Dia da semana manhã: "+str(normal_day))
    #print("\n")
    #print("Dia da semana tarde: "+str(normal_night))
    #print("\n")
    #print("Fim de semana manhã: "+str(weekend_day))
    #print("\n")
    #print("Fim de semana tarde: "+str(weekend_night))

    return {"week":[normal_day,normal_night],"weekend":[weekend_day,weekend_night],"index":""}

#data_arr_dep = gen_all()


############## (TODO TODO) SIMULATED FUNCTION WITH RANDOM DATA (TODO TODO) ##############
# Traffic ?
def gen_traffic(weekend,time_taken):
    arr = traffic_week
    if(weekend):
        arr=traffic_weekend
    random.seed(time.time_ns())
    traffic_main = random.choices(
        population = [h[0] for h in arr], 
        weights = [h[1] for h in arr], k=1)
    gen = arr[[h[0] for h in arr].index(traffic_main[0])]

    time_percentage_plus = random.randint(gen[2]*100,gen[3]*100)
    time_taken = time_taken + time_taken * (time_percentage_plus/100)

    return trunk_to_time(time_taken)

        
    

# Calculate time based on dist and velocity
def calculate_time(arr):
    random.seed(time.time_ns())
    dist_home_main = random.choices(
        population = [h[0] for h in arr], 
        weights = [h[1] for h in arr], k=1)
    gen = arr[[h[0] for h in arr].index(dist_home_main[0])]
    
    dist = round(random.SystemRandom().uniform(gen[2], gen[3]),2)
    vel = random.randint(gen[4],gen[5])
    #print(gen)
    #print(f"Vel: {vel} ; Dist: {dist}")
    return {"dist":dist,"time":round(dist / vel,2)}

# Time to time stamp
def trunk_to_time(numero):
    trunk_multiple = TRUNK_MULTIPLE # minutos time stamp em horas
    multiplo_anterior = (numero // trunk_multiple) * trunk_multiple
    multiplo_posterior = multiplo_anterior + trunk_multiple
    if abs(numero - multiplo_anterior) < abs(numero - multiplo_posterior):
        if(multiplo_anterior < trunk_multiple+0.01):
            multiplo_anterior = trunk_multiple
        return multiplo_anterior
    else:
        if(multiplo_posterior < trunk_multiple+0.01):
            multiplo_posterior = trunk_multiple
        return multiplo_posterior
    
def energy_spent(dist,index):
    #random.seed(time.time_ns())
    #min = energy["spend"][0]
    #max = energy["spend"][1]
    #spend_per_km = round(random.SystemRandom().uniform(min, max),2)
    spend_per_km = charger_ex[index][3]
    
    #print(spend_per_km)
    spent = round((dist * spend_per_km)/100,2)
    return spent*2

def work_charge(chance):
    random.seed(time.time_ns())
    return random.randint(1,100)>(100-100*chance)

def get_rand_index(arr):
    random.seed(time.time_ns())
    return random.randint(0,len(arr)-1)
def charge():
    random.seed(time.time_ns())
    num = random.randint(1,100)
    if(num > 90): num = random.randint(90,100)
    elif(num > 80 and num <= 90): num = random.randint(60,80)
    else: num = random.randint(80,90)
    return num

def get_predicted_energy(spent,work,index):
    max_bat = charger_ex[index][2]
    
    if(work):
        spent=spent/2
    lost_percentage = round(spent / max_bat,2)*100
    return lost_percentage
    

############## ############## ############## ############## ############## ##############
def test_sim(dia_atual,dia_semana,data_arr_dep,trip_work_time,type,previous1,previous2,dist,regular,previous_energy,other,test,charger,previous_charge,data_arr_dep2,type_week,index):

    leave_home = data_arr_dep[type][0]
    leave_home2 = data_arr_dep2[type_week][0]
    

    arrive_home = data_arr_dep[type][1]
    soc_x = -1
    trigger=0
    soc_y = -1
    last = -2

    #print(str(trip_work_time))

    multiplo = -0.25
    charging = -1
    charger_this=-1
    arr = []
    while multiplo <= 23.50:
        random.seed(time.time_ns())
        multiplo += 0.25
        if(trigger>0):
            leave_home = leave_home2
        if (leave_home>arrive_home):
            if(arrive_home <= multiplo and leave_home >= multiplo):
                charging = 0
                
            elif (arrive_home-trip_work_time <= multiplo and leave_home >= multiplo):
                if(previous2<1):
                    
                    work = False
                    if(regular):
                        work = work_charge(0.1)
                    lost_percentage = get_predicted_energy(energy_spent(dist,index),work,index)
                    soc_y = previous_energy-lost_percentage
                    previous2=previous2+1
                    soc_x=-1
                    charger_this = charger
                charging = 1
                trigger=1
            else:
                
                charger_this=-1
                soc_x=-1
                soc_y=-1
                charging = -1
                
        else:
            if(leave_home >= multiplo or arrive_home <= multiplo):
                charging = 0
                
                
            elif (arrive_home-trip_work_time <= multiplo):
                if(previous2<1):
                    work = False
                    if(regular):
                        work = work_charge(0.1)
                    lost_percentage = get_predicted_energy(energy_spent(dist,index),work,index)
                    soc_y = previous_energy-lost_percentage
                    soc_x=-1
                    charger_this = charger
                    previous2=previous2+1
                charging = 1
                trigger=1
                
            else:
                
                charger_this=-1
                soc_x=-1
                soc_y=-1 
                charging = -1

                

        if(leave_home<0):
            trip_work_time = 0
            charging = 0
            soc_y=-1

        # Departure Socket  
        
        if(last == 1 and charging == 0 ):
            previous1=0
        if(other==0 and charging == 0):
            #print(test)
            soc_x=test
            charger_this = previous_charge
            
        else:
            other=charging
            if(charging==0 and previous1<1):
                soc_y=-1
                soc_x = charge()
                charger_this = charger
                previous1=previous1+1
        last = charging
        
        if(soc_x>0):
            test = soc_x
        if(previous_charge==-2):
            charger_this=charger

        if(charger_this>=0):
            previous_charge=charger_this
            
        
        if(soc_y<-1):
            soc_y = abs(soc_y)
        ##########################
        #print("[+] Time: " + str(multiplo) + " State: " + convert_to_string(charging) + " Leave Home: " + str(leave_home) + " Arrive Home: " + str(arrive_home) + " Trip time: "+ str(trip_work_time))
        new_element = [dia_atual,dia_semana,multiplo,leave_home,arrive_home,trip_work_time,convert_to_string(charging),int(soc_y),soc_x,charger_this]
        
        
        
        #print(new_element)
        arr.append(new_element)
    return {"arr":arr,"previous1":previous1,"previous2":previous2,"charged":soc_x,"last":charging,"sock":test,"charge":previous_charge}

def convert_to_string(num):
    if(num==0): return "Charging"
    elif(num==-1): return "NoState"
    else: return "Incoming"

def convert_to_number_charging(string):
    if(string=="Charging"):return 1
    elif (string=="Incoming"):return 2
    else:return 3

# Hipotese de ter uma rotina diferente na semana # +PODE SER USADO PARA HIPÓTESES
def rotina_change(rotin_chance):
    random.seed(time.time_ns())
    max = 100
    min = 1
    chance = max - (rotin_chance*max)
    num = random.randint(min,max)
    return  num > chance

def new_gen(arr):
    charger_constant = get_rand_index(arr)
    while(arr[charger_constant]==""):
        charger_constant = get_rand_index(arr)
    return charger_constant

def monthly_gen(ano,mes,charg):
    gen_time1 = calculate_time(dist_home)
    gen_time2 = calculate_time(dist_home_weekend)
    rando_cena_dia = get_random_cena(semana_dia)
    rando_cena_noite = get_random_cena(semana_noite)
    choice = get_random_cena(fimdesemana)
    previous_energy = 80
    work = work_charge(0.3)
    other = -2
    test = -1
    previous_charge = -2
    trip_work_time_constant_week = round(trunk_to_time(gen_time1["time"]),2)
    trip_work_time_constant_weekend = round(trunk_to_time(gen_time2["time"]),2)
    charger_constant = charg

    
    #charger_usable[charger_constant] = ""
    primeiro_dia = datetime.date(ano, mes, 1)

    ultimo_ano = ano+YEARS



    # Obter o último dia do mês
    if mes == 12:
        proximo_mes = 1
        proximo_ano = ultimo_ano + 1
    else:
        proximo_mes = mes + 1
        proximo_ano = ultimo_ano
    ultimo_dia_mes_atual = datetime.date(proximo_ano, proximo_mes, 1) - datetime.timedelta(days=1)

    dia_atual = primeiro_dia
    #dia_atual = ultimo_dia_mes_atual
    arr = []
    week = None
    week2 = gen_all(rando_cena_dia,rando_cena_noite,choice)
    
    while dia_atual <= ultimo_dia_mes_atual:
        #print(dia_atual)
        previous1=0
        previous2=0
        if(week==None):
            week = gen_all(rando_cena_dia,rando_cena_noite,choice)
        else:
            week = week2

        week2 = gen_all(rando_cena_dia,rando_cena_noite,choice)

        
        trip_work_time_week = trip_work_time_constant_week
        
        trip_work_time_weekend = trip_work_time_constant_weekend
        charger = charger_constant
        dia_semana = dia_atual.isoweekday()  # Nome do dia da semana
        type_week=""
        if(dia_semana+1>=6 and dia_semana+1<=7):
            type_week = "weekend"
        else: type_week="week"
        dia_mes = dia_atual.day  # Dia do mês
        if(rotina_change(CHARGER_CHANGE)):
            charger = charg
        # Verificar se é fim de semana (sábado ou domingo)
        if dia_atual.weekday() >= 5:
            if(rotina_change(ROTINA_CHANGE)):
                gen_time1 = calculate_time(dist_home)
                
                trip_work_time_week = round(trunk_to_time(gen_time1["time"]),2)
            trip_work_time_week = gen_traffic(False,trip_work_time_week)
            gen = test_sim(dia_atual,dia_semana,week,trip_work_time_week,"weekend",previous1,previous2,gen_time1["dist"],work,previous_energy,other,test,charger,previous_charge,week2,type_week,charg)
            previous1 = gen["previous1"]
            previous2 = gen["previous2"]
            other = gen["last"]
            test = gen["sock"]
            previous_charge = gen["charge"]
            #print(str(other)+"HELLO")
            arr.extend(gen["arr"])
            if(gen["charged"]>0):
                previous_energy = gen["charged"]
            #print(f"{dia_mes} Fim de semana, {dia_semana}")
        else:
            if(rotina_change(0.6)):
                gen_time2 = calculate_time(dist_home_weekend)
                trip_work_time_weekend = round(trunk_to_time(gen_time2["time"]),2)
            trip_work_time_weekend = gen_traffic(True,trip_work_time_weekend)
            gen = test_sim(dia_atual,dia_semana,week,trip_work_time_weekend,"week",previous1,previous2,gen_time2["dist"],work,previous_energy,other,test,charger,previous_charge,week2,type_week,charg)
            previous1 = gen["previous1"]
            previous2 = gen["previous2"]
            other = gen["last"]
            test = gen["sock"]
            previous_charge = gen["charge"]
            #print(str(other)+"HELLO")
            arr.extend(gen["arr"])
            if(gen["charged"]>0):
                previous_energy = gen["charged"]
            #print(f"{dia_mes} Dia de semana, {dia_semana}")
        
        dia_atual += datetime.timedelta(days=1)
    return arr

def can_write(hour1,hour2):
    hour_1 = str(hour1).split(".")[0]
    hour_2 = str(hour2).split(".")[0]
    return hour_1!=hour_2


def choose_minute(arr):
    favourite = ""
    nostate = 0
    for data in arr:
        data_fix = data.split(",")
        if(data_fix[7]!=""):
            favourite = data
            return favourite
    for data in arr:
        data_fix = data.split(",")
        if(data_fix[4]!=""):
            favourite = data
            return favourite
    if (favourite==""):
        for data in arr:
            data_fix = data.split(",")
            if(data_fix[4]==""):
                nostate = nostate + 1 
        for data in arr:
            data_fix = data.split(",")
            if(nostate>=2):
                if(data_fix[4]==""):
                    favourite=data
            else:
                if(data_fix[4]!=""):
                    favourite=data
    
    return favourite

def arredonda_tempos(tempo):
    decimal = tempo - int(tempo)
    if decimal == 0.75:
        return int(tempo) + 1
    else:
        return tempo
    
def convert_format(arr,index):
    #[dia_atual,dia_semana,multiplo,leave_home,arrive_home,trip_work_time,convert_to_string(charging),soc_y,soc_x,charger_this]
    #Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival 
    separator = ","
    with open("C:\\Users\\steam\\Desktop\\Opeva\\EV"+charger_ex[int(index)][0]+'.csv', 'w') as file:
    # Write multiple lines
        header = "Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival"
        file.write(header+"\n")
        charge_arr = []
        previous_hour = -1
        write = False
        for data in arr:

            hour = str(data[2]) # Hour
            if(float(previous_hour)>=0):
                write = can_write(hour,previous_hour)
            month = str(data[0].month) # Month para format do date time
            
            

            day_type =  str(data[0].weekday() + 1) # Week day
            state = str(convert_to_number_charging(data[6])) # Charging state
            dest_charger = data[9] # Number of index charger arr
            dest_charger_name = "" # Charger name
            if(dest_charger>=0): # If Charger not assigned
                dest_charger_name=str(charger_ex[dest_charger][1][0])
            
            departure_time = str(data[3]) # Departure Time 
            
            #print(departure_time.split(".")[0]+"     "+hour.split(".")[0])
            #time.sleep(2)
            #print(departure_time.split(".")[0])
            if(int(float(departure_time.split(".")[0]))<0):
                departure_time=""
            elif(state == "1"):
                if(int(departure_time.split(".")[0])<=0):
                    departure_time_t = (24-int(hour.split(".")[0])) + int(departure_time.split(".")[0])
                   #print("1-type- "+str(departure_time)+" hour- "+hour.split(".")[0])
                elif(int(departure_time.split(".")[0]) < int(hour.split(".")[0])):
                    departure_time_t = (24-int(hour.split(".")[0])) + int(departure_time.split(".")[0])
                    #print("2-type- "+str(departure_time)+" hour- "+hour.split(".")[0])
                else:
                    departure_time_t = int(departure_time.split(".")[0])-int(hour.split(".")[0])
                    #print("3-type- "+str(departure_time)+" hour- "+hour.split(".")[0])
                
                #print("--")
                #print(str(departure_time.split(".")))
                #print(str(hour.split(".")))
                #print(state)

                departure_time = departure_time_t
            else: 
                departure_time = ""
                #print("4-type-"+str(departure_time)+" hour-"+hour.split(".")[0])
            
            #time.sleep(2)
            soc_departure = data[8]
            soc_departure_name = "" # Soc Departure
            if(soc_departure>=0):
                soc_departure_name = str(soc_departure)
            arrival_time = str(data[4]) # Arrival Time
            soc_arrival = data[7] 
            
            if(int(arrival_time.split(".")[0])<0):
                arrival_time=""
            elif(state=="2"):
                #print("--")
                #print(str(arrival_time.split(".")))
                #print(str(hour.split(".")))
                #print(state)
                #print("###############")
                arrival_time_t = int(arrival_time.split(".")[0])-int(hour.split(".")[0])
                if(float(arrival_time)-int(float(arrival_time))==0.0):
                    arrival_time_t = arrival_time_t-1
                elif(float(arrival_time)-arredonda_tempos(float(hour)+0.25)==0.0):
                    arrival_time_t=0
                arrival_time = arrival_time_t
                
                
            else: arrival_time = ""
            soc_arrival_name = "" # Soc Arrival
            if(soc_arrival>=0):
                soc_arrival_name = str(soc_arrival)

        
            
            writer_line = month+separator+str(hour).split(".")[0]+separator+day_type+separator+state+separator+dest_charger_name+separator+str(departure_time).split(".")[0]+separator+soc_departure_name+separator+str(arrival_time).split(".")[0]+separator+soc_arrival_name
            if(write):
                file.write(choose_minute(charge_arr)+"\n")
                #for x in charge_arr:
                    #print(x)
                #print("Choosen->" + str(choose_minute(charge_arr)))
                charge_arr=[]
                
                #print("##############")

            charge_arr.append(writer_line)
            previous_hour = hour
            
    
charger_usable = charger_ex[:]  
for x in range(len(charger_ex)):
    arr = monthly_gen(2023,7,x)
    data = ["Current Date", "Day Name","Time","Leave Home","Arrive Home","Trip Time","State","Arrival Soc","Departure Soc","Charger"]
    df = pd.DataFrame(arr, columns=data)
    filename = 'C:\\Users\\steam\\Desktop\\Opeva\\EV-'+charger_ex[x][0]+'.csv'
    df.to_csv(filename, index=False)
    print(f"Data successfully written to {filename}")
    convert_format(arr,str(x))







