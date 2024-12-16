import random
import time
import datetime
import pandas as pd 
import numpy as np
import math
import copy
from scipy.stats import skewnorm, truncnorm
import matplotlib.pyplot as plt



################## Global ##################

CHANGE_ROUTINE_WEEK = 0.01
CHANGE_ROUTINE_WEEKEND = 0.01
WORK_WEEKEND_CONSTANT = 0.01
WORK_WEEKEND_RAND_1 = 0.01
WORK_WEEKEND_RAND_2 = 0.01
CHARGE_AT_WORK=1
CHARGE_THRESHOLD=95
CHARGE_MIN_THRESHOLD = 10
YEARS=4

CHARGE_BAT = [15,85]

semana_dia = [ # Prob ; Min Left ; Max Left 
    ("Normal1", 0.48,7,8), # 7-8  
    ("Normal2", 0.50,8,9), # 8-9  
    ("Cedo", 0.01,5,6), # 5-6  
    ("tarde", 0.01,9,14), # 9-12   
]
semana_noite = [ # Prob ; Min Arrive ; Max Arrive 
    ("Normal1", 0.40,17,18), # 17-18
    ("Normal2", 0.40,18,19), # 18-19
    ("tarde", 0.10,20,23), # 20-22  
    ("cedo", 0.10,15,16), # 14-16
]

cars_all = [ # Name ; Max ; Usage, Pre data
    ("ev_id_7", 85,19,[]),
    ("ev_id_8", 70,17,[]),
    ("ev_id_9", 60,16,[]),
    ("ev_id_10", 80,19,[]),
]

# Store Dataset
store = {"ev_id_7":[],"ev_id_8":[],"ev_id_9":[],"ev_id_10":[]}

cars_routine = []

dist_home = [ # Prob ; Min dist ; Max dist ; Min vel ; Max vel
    ("close", 0.35,6,10,30,50), 
    ("medium", 0.4,10,30,40,50), 
    ("far", 0.25,40,100,60,80),
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

cars_in_use = [ #array to track what car is in use
]

chargers_all = [ # Name ; Leave Hour ; Charge
    ["charger_1_1",-1,12,],
    ["charger_1_2",-1,10,],
    ["charger_1_3",-1,12,],
    ["charger_1_4",-1,10,], 
]

chargers_in_use = [ # array to track what charger is in use
]
###########################################################

#################### Utilitários Gerais ###################


def get_random_cena(arr,volt): # Função usada para gerar as cenas aleatórias com o index do mesmo
    # Seed variando com o tempo
    random.seed(time.time_ns()*volt)
    cena_aleatoria = random.choices(
        population = [h[0] for h in arr], 
        weights = [h[1] for h in arr], k=1)
    return {"name":cena_aleatoria[0],"index":[h[0] for h in arr].index(cena_aleatoria[0])}

def change_prob(prob): # Função que dando uma probabilidade diz se ocorreu ou não
    system_random = random.SystemRandom()
    system_random.seed(time.time_ns())
    probabilidade = max(0.01, min(1.0, prob))
    
    numero_aleatorio = system_random.random()
    return round(numero_aleatorio, 2) <= probabilidade

# BACKUP METHOD #TODO
'''def random_between(min,max): # Random between two numbers
    system_random = random.SystemRandom()
    system_random.seed(time.time_ns())
    return system_random.uniform(min, max)'''

def random_between(min_value,max_value): # Random between two numbers
    skew=1
    size=1
    std = 0.5
    mean = (min_value + max_value) / 2
    # Convertendo a medida da assimetria para o parâmetro da distribuição
    a = skew / np.sqrt(1 + skew**2)

    # Gerando valores da distribuição
    values = skewnorm.rvs(a, loc=mean, scale=std, size=size)

    # Aplicando os limites
    clipped_values = np.clip(values, min_value, max_value)
    
    return clipped_values[0]

def check_available(all,in_use): # Função para saber quais estão disponíveis
    return [sub for sub in all if sub[0] not in [item[0] for item in in_use]]

def random_arr_element(arr): # Função para ir buscar elemento de array aleatóriamente com os mesmos pesos para todos
    random.seed(time.time_ns())
    return random.choice(arr)

def get_first_last_day(ano,mes): # Função para ir buscar o primeiro e ultimo dia do mes
    ultimo_ano = ano+YEARS
    primeiro_dia = datetime.date(ano, mes, 1)
    # Obter o último dia do mês
    if mes == 12:
        proximo_mes = 1
        proximo_ano = ultimo_ano + 1
    else:
        proximo_mes = mes + 1
        proximo_ano = ultimo_ano
    ultimo_dia_mes_atual = datetime.date(proximo_ano, proximo_mes, 1) - datetime.timedelta(days=1)

    dia_atual = primeiro_dia
    return {"dia_atual":dia_atual,"dia_ultimo":ultimo_dia_mes_atual}

def find_available_element(all,in_use): # Combine 2 utilitários para trocar de arrays geral para em uso e obter elemento novo
    if(len(all)==len(in_use)):return None
    available_elements = check_available(all,in_use)
    element = random_arr_element(available_elements)
    in_use.append(element)
    return element

def convert_perfect(number): # Função que permite colocar tempo do 0.15 a 1 hora
    return math.ceil(number)

def calc_trip_time(vel,dist,traffic): # Função para calcular quanto demora a chegar ao trabalho tendo em conta velocidade, distância e trânsito
    time_no_extra = dist/vel 
    time_extra = time_no_extra + (time_no_extra*traffic)
    return time_extra

# Encontra o minimo baseado no index
def get_min(arr,index):
    cont = 99999
    ele=None
    for x in arr:
        value = x[index]
        if value < cont:
            ele = x
            cont = value
    return ele

# Just add it the data to our store
def find_element_add_data(element_name,store,data):
    store[element_name].extend(data)

# Quanto gastei de energia do carro
def spent_bat(element,dist):
    cap_max = element[1]
    cons_avg = element[2]
    bat_spent = round(cons_avg * dist/100,2)
    perc_spent = round((bat_spent/cap_max *100),2)
    return {"Espent":min(bat_spent, cap_max),"Pspent":perc_spent,"Eleft":(cap_max-bat_spent),"Pleft":(100-perc_spent)}  # Limita o resultado à capacidade máxima da bateria
###########################################################

# Vamos criar as rotinas constantes iniciais
def pre_routine_fill(store):
    for x in cars_all:
        routine_week_arrive_constant = get_random_cena(semana_dia,len("routine_week_arrive_constant")) # chega trabalho semana C
        routine_week_leave_constant = get_random_cena(semana_noite,len("routine_week_leave_constant")) # sai trabalho semana C
        distance_home_week_contstant = get_random_cena(dist_home,len("distance_home_week_contstant")) # distancia trabalho semana C
        weekend_work_constant = change_prob(WORK_WEEKEND_CONSTANT) # Normal trabalhar ao sábado ? C
        x[3].extend([routine_week_arrive_constant,routine_week_leave_constant,distance_home_week_contstant,weekend_work_constant])
        # 0 -> week arrive work routine semana_dia[?]
        # 1 -> week leave work routine semana_noite[?]
        # 2 -> distance work dist_home[?]
        # 3 -> Work in Weekends
        time.sleep(0.1) # Make some changes when gen

# Main Function
def routine():
    global cars_in_use
    global cars_all
    global chargers_in_use
    global chargers_all
    global store
    ano = 2023 ## TODO Change
    mes = 8 ## TODO Change
    pre_routine_fill(store) # Criar rotinas constantes
    dic_month = get_first_last_day(ano,mes)
    dia_atual = dic_month["dia_atual"]
    ultimo_dia_mes_atual = dic_month["dia_ultimo"]
    
    while dia_atual <= ultimo_dia_mes_atual:
        element = find_available_element(cars_all,cars_in_use)
        day_type = dia_atual.weekday() + 1
        while(element!=None):
            weekend_work_constant = element[3][3] # Trabalha fim de semana regularmente?
            
            if dia_atual.weekday() == 5: # Weekend Saturday
                weekend_work = change_prob(WORK_WEEKEND_RAND_1) # Trabalha Random Neste Sábado?
                data = day_pre(element,weekend_work_constant or weekend_work,True,False,mes,day_type)
                
            elif dia_atual.weekday() == 6: # Weekend Sunday
                weekend_work = change_prob(WORK_WEEKEND_RAND_2) # Trabalha Random Neste Domingo?
                data = day_pre(element,weekend_work,True,False,mes,day_type)
                
            else: # Week
                routine_change = change_prob(CHANGE_ROUTINE_WEEK)
                
                data = day_pre(element,True,False,routine_change,mes,day_type)
            #print(element)
            find_element_add_data(element[0],store,data)
            element = find_available_element(cars_all,cars_in_use)
            #print(chargers_all)
        #print(dia_atual)
        #print("###############")
        
        dia_atual += datetime.timedelta(days=1)
        cars_in_use = []
        chargers_in_use = []
        #print("##########")

# So how much do i need to charge? and more info
def left_charge(arrive_bat,energy_spent_trip,max_bat_cap,charger_value):

    value_to_charge = energy_spent_trip * 2 # Energy i want to charge 2 times the value i spent to home or to work
    
    ideal_bat =  value_to_charge + arrive_bat # I want to leave with this charge but i can pass the max capaity!!!
    max_charge = max_bat_cap*(CHARGE_THRESHOLD/100) # Max Charge until this value 
    new_bat = min(ideal_bat,max_charge) # My new Bat value the ideal or max bat value
    if(new_bat==max_charge):
        value_to_charge = max_charge-arrive_bat
    time_spent_charging = convert_perfect(value_to_charge / charger_value[2]) # So i will spend this amount of time charging but lets convert it to hours
    
    return {"Ebatery":round(value_to_charge,2),"time":time_spent_charging,"charger":charger_value}

# OK, i want to charge ? Do i get a charger ? But if not how much energy would i need ? Add the intention to charge if wanted !!
def avaliate_charge(element,dist,arrive_number):
    global chargers_in_use
    global chargers_all
    bat_dict = spent_bat(element,dist)
    bat_arrive = convert_perfect(random_between(CHARGE_BAT[0],CHARGE_BAT[1]))
    
    charger=None
    dict_charge_info=None
    will_charge = change_prob(CHARGE_AT_WORK)

    # If battery after 2 times the trip is below 10 percent just charge it  
    if (bat_arrive - bat_dict["Pspent"]*2) <= 10: will_charge = True

    # We have premission to charge
    if(will_charge):
        # Do we have a charger available?
        charger = find_available_element(chargers_all,chargers_in_use)
        charger_to_use=charger
        
    if(will_charge and charger==None):
        charger_to_use = get_min(chargers_in_use,1)  
        dict_charge_info = left_charge(element[1]*(bat_arrive/100),bat_dict["Espent"],element[1],charger_to_use)
    # Did we get a charger
    if(charger!=None):
        dict_charge_info = left_charge(element[1]*(bat_arrive/100),bat_dict["Espent"],element[1],charger_to_use)
        
        chargers_in_use[len(chargers_in_use)-1][1] = convert_perfect(arrive_number + dict_charge_info["time"])
    else:
        pass
        #print(element[0])
        

    # Even if we dont get a charger it can be available in a specific hour so we get the intention to charge
    return {"charge":will_charge,"charger":charger,"info":dict_charge_info,"arrive":bat_arrive,"max":element[1]} 


# Day pre configuration steps
def day_pre(element,work,weekend,routine_change,month,day_type):
   
    if(work):
        arr = traffic_week
        if weekend: arr = traffic_weekend
        traffic_arrive = get_random_cena(arr,len("traffic_arrive")) # Tipo de trânsito no dia
        routine_arrive = element[3][0]
        leave_routine = element[3][1]
        dist_work = element[3][2]
        
        if(routine_change):
            routine_arrive = get_random_cena(semana_dia,len("routine_arrive"))
            leave_routine = get_random_cena(semana_noite,len("leave_routine"))

        traffic_arrive_number = random_between(arr[traffic_arrive["index"]][2],arr[traffic_arrive["index"]][3]) # Percentagem que acrescenta ao tempo de viagem
        dist_work_number = random_between(dist_home[dist_work["index"]][2],dist_home[dist_work["index"]][3]) # Distância de trabalho Km
        vel_number = random_between(dist_home[dist_work["index"]][4],dist_home[dist_work["index"]][5]) # Velocidade média Km/h
        arrive_number = round(random_between(semana_dia[routine_arrive["index"]][2],semana_dia[routine_arrive["index"]][3]),0) # hora chegada
        leave_number = round(random_between(semana_noite[leave_routine["index"]][2],semana_noite[leave_routine["index"]][3]),0) # hora saida
        
        real_time = convert_perfect(calc_trip_time(vel_number,dist_work_number,traffic_arrive_number)) # Temos o Tempo da Trip em Horas já
        dict_charge = avaliate_charge(element,dist_work_number,arrive_number)
        
        #print(dict_charge)
        return day_gen(arrive_number,leave_number,real_time,dict_charge,month,day_type,dict_charge["arrive"])
    else:
        return day_gen(None,None,None,None,month,day_type,None)

# Main function for daily gen
def day_gen(arrive_hour,leave_hour,trip_time,dict_charge,month,day_type,arrive_soc):
    global chargers_in_use
    info_charge = None
    charger = None
    arrive_hour_charger=None
    wait_charger = False
    charged = None
    if(arrive_hour==None or dict_charge==None): # No itention to Charge fill with -1
        want_to_charge = False
    else:
        want_to_charge = dict_charge["charge"] # Work but not interested in Charging fill with -1
        info_charge = dict_charge["info"]
        charger = dict_charge["charger"]

    # Yes i want to charge
    if(want_to_charge and charger!=None): # Yes i have a charge
        leave_hour = leave_hour#min(info_charge["time"]+arrive_hour+1,leave_hour)
        arrive_hour_charger = arrive_hour
        charged = info_charge["Ebatery"]
    elif(want_to_charge and charger==None): # I can try to charge but i will have to wait for available charge
        arrive_hour_charger = info_charge["charger"][1]+1
        index_arr = chargers_in_use.index(info_charge["charger"])
        leave_hour = leave_hour#min(arrive_hour_charger+info_charge["time"],leave_hour) # Lets see if leave hour is greater than the after charge
        chargers_in_use[index_arr][1]=arrive_hour_charger # Updating the value in chargers in use
        max_value = info_charge["time"] * info_charge["charger"][2]
        if(leave_hour<=arrive_hour_charger+info_charge["time"]):
            charged = info_charge["Ebatery"]
        else:
            charged = round(random_between(max_value-info_charge["charger"][2]+1,max_value),2)

    multiplo=0
    string_data=""
    separator = ","
    data_arr = []
    while multiplo <= 23:
        state = "3"
        string_data = str(month)+separator+str(multiplo)+separator+str(day_type)+separator
        if(not want_to_charge):
            string_data = string_data + state + separator+separator+separator+separator+separator+"\n"
        else:
            incoming = ""
            socket_arr = ""
            charging = ""
            socket_dep = ""
            start_incoming = trip_time + (arrive_hour_charger - arrive_hour)
            if( multiplo <= arrive_hour_charger and multiplo >= arrive_hour_charger-start_incoming):
                incoming = round(arrive_hour_charger-multiplo,0)
                state = "2"
                string_data = string_data + state + separator+info_charge["charger"][0]+separator+separator+separator+str(incoming).split(".")[0]+separator+str(arrive_soc)+"\n"
            elif( multiplo > arrive_hour_charger and multiplo <= leave_hour):
                charging = round(leave_hour-multiplo,0)
                state = "1"
                departure_soc = round((charged/dict_charge["max"])*100+arrive_soc,0)
                string_data = string_data + state + separator+info_charge["charger"][0]+separator+str(charging).split(".")[0]+separator+str(departure_soc).split(".")[0]+separator+separator+"\n"
            else:
                string_data = string_data + state + separator+separator+separator+separator+separator+"\n"

        multiplo+=1
        data_arr.append(string_data)
    
    return data_arr




# lets fill the files
def to_file():
    with open("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\"+'ev_id_10.csv', 'w') as file:
        header = "Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival"
        file.write(header+"\n")
        for x in store["ev_id_10"]:
            file.write(x)

    with open("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\"+'ev_id_9.csv', 'w') as file:
        header = "Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival"
        file.write(header+"\n")
        for x in store["ev_id_9"]:
            file.write(x)
            
    with open("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\"+'ev_id_8.csv', 'w') as file:
        header = "Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival"
        file.write(header+"\n")
        for x in store["ev_id_8"]:
            file.write(x)

    with open("C:\\Users\\steam\\Desktop\\Bolsa\\Opeva\\"+'ev_id_7.csv', 'w') as file:
        header = "Month,Hour,Day Type,State,Destination Charger,Estimated Departure Time,Required Soc At Departure, Estimated Arrival Time, Estimated SOC at Arrival"
        file.write(header+"\n")
        for x in store["ev_id_7"]:
            file.write(x)
        

def main():
    # Start main step
    routine()
    # Fill files
    to_file()

main()



'''def asymmetric_normal(mean, std, skewness, min_value, max_value, size=1):
    # Convertendo a medida da assimetria para o parâmetro da distribuição
    a = skewness / np.sqrt(1 + skewness**2)

    # Gerando valores da distribuição
    values = skewnorm.rvs(a, loc=mean, scale=std, size=size)

    # Aplicando os limites
    clipped_values = np.clip(values, min_value, max_value)

    return clipped_values

# Exemplo de uso
mean = 8.0
std = 0.5
skewness = -10
min_value = 0.0
max_value = 15.0


# Numero média # Desvio padrão # Valor de assimetria para se mais á direita ou esquerda # Min value # Max value # Size amostra
amostra = asymmetric_normal(mean, std, skewness,min_value,max_value,size=1000) 


print(amostra)

plt.hist(amostra, bins=30, density=True, alpha=0.7)
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma da distribuição gerada por left_dist')
plt.show()
'''