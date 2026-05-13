import pygame
import socket
import sys
import os
import time
import csv
import numpy as np

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# LOGICA F1: 
# Fino a 1.3 sei sul cordolo (OK)
# Oltre 1.3 sei fuori pista (RESTART + GIRO SPORCO)
TRACK_LIMIT = 1.3 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Mapping DualShock 4
AXIS_STEER = 0
AXIS_ACCEL = 5
AXIS_BRAKE = 4

class ServerState():
    def __init__(self):
        self.d = dict()
    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = self.destringify(w[1:])
    def destringify(self, s):
        if not s: return s
        if type(s) is str:
            try: return float(s)
            except ValueError: return s
        elif type(s) is list:
            if len(s) < 2: return self.destringify(s[0])
            else: return [self.destringify(i) for i in s]

class DriverAction():
    def __init__(self):
        self.d = {'accel': 0, 'brake': 0, 'clutch': 0, 'gear': 1, 'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

def get_joystick_input(joystick, current_speed):
    pygame.event.pump()
    raw_steer = -joystick.get_axis(AXIS_STEER)
    if abs(raw_steer) < 0.02:
        steer = 0.0
    else:
        steer = np.sign(raw_steer) * (abs(raw_steer) ** 2.0)
        if current_speed > 50:
            steer *= max(0.4, 1.0 - (current_speed - 50) / 300.0)
    accel = (joystick.get_axis(AXIS_ACCEL) + 1.0) / 2.0
    brake = (joystick.get_axis(AXIS_BRAKE) + 1.0) / 2.0
    return steer, accel, brake

def save_to_disk(buffer, headers, lap_number, lap_time):
    if not buffer: return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    minutes = int(lap_time // 60)
    seconds = int(lap_time % 60)
    milliseconds = int((lap_time % 1) * 1000)
    time_str = f"{minutes:02d}-{seconds:02d}-{milliseconds:03d}"
    filename = f"lap_{lap_number:03d}_time_{time_str}_{timestamp}.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(buffer)
    print(f"\n>>> [SALVATO] Giro {lap_number} | Tempo: {time_str}")

def manual_recording():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERRORE: Collega il controller!")
        return
    js = pygame.joystick.Joystick(0)
    js.init()

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"
    so.sendto(initmsg.encode(), (HOST, PORT))

    while True:
        try:
            sockdata, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in sockdata.decode():
                print(">>> SISTEMA PRONTO. GUIDA SUI CORDOLI MA EVITA L'ERBA!")
                break
        except:
            so.sendto(initmsg.encode(), (HOST, PORT))

    S = ServerState()
    R = DriverAction()
    KEYS_TO_IGNORE = ['opponents', 'focus', 'fuel', 'damage', 'z', 'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced', 'racePos']

    lap_buffer = []
    prev_lap_time = 0.0
    initial_damage = None
    is_dirty = False
    headers = None
    t0 = time.time()
    lap_counter = 0
    waiting_for_restart = False

    try:
        while True:
            try:
                sockdata, _ = so.recvfrom(DATA_SIZE)
                sockstr = sockdata.decode()
                
                if '***restart***' in sockstr:
                    print("\n[RESET] Gara riavviata.")
                    lap_buffer, is_dirty, prev_lap_time = [], False, 0.0
                    initial_damage = None
                    R.d['meta'] = 0
                    waiting_for_restart = False
                    continue
                
                S.parse_server_str(sockstr)
            except: continue

            if waiting_for_restart:
                so.sendto(repr(R).encode(), (HOST, PORT))
                continue

            if initial_damage is None:
                initial_damage = S.d.get('damage', 0)

            # --- LOGICA F1: CORDOLO VS FUORI PISTA ---
            track_pos = abs(S.d.get('trackPos', 0))
            
            # Se sei tra 1.0 e 1.3 sei sul cordolo -> NON facciamo nulla, il giro resta pulito.
            
            # Se superi il limite del cordolo (es. 1.3) -> Giro SPORCO + RESTART
            if track_pos > TRACK_LIMIT:
                print(f"\n[!!!] FUORI PISTA ({track_pos:.2f}). Giro annullato. Reset...")
                is_dirty = True # Marca come sporco
                R.d['meta'] = 1
                waiting_for_restart = True
                so.sendto(repr(R).encode(), (HOST, PORT))
                continue

            # --- FINE GIRO ---
            cur_time = S.d.get('curLapTime', 0)
            if cur_time < prev_lap_time and prev_lap_time > 10.0:
                last_lap_time = S.d.get('lastLapTime', 0)
                # Salva solo se non hai preso danni e non sei uscito oltre il cordolo
                if not is_dirty and len(lap_buffer) > 500:
                    lap_counter += 1
                    save_to_disk(lap_buffer, headers, lap_counter, last_lap_time)
                else:
                    print(f">>> [SCARTATO] Giro non valido (Danni o Fuori Pista).")
                
                lap_buffer, is_dirty = [], False
                initial_damage = S.d.get('damage', 0)

            prev_lap_time = cur_time

            # Controllo Danni (se colpisci un muro, il giro è sporco)
            if (S.d.get('damage', 0) - initial_damage) > 1.0:
                if not is_dirty:
                    print("\n[!] DANNI RILEVATI - Giro sporco.")
                is_dirty = True

            # --- CONTROLLI ---
            speed = S.d.get('speedX', 0)
            steer, accel, brake = get_joystick_input(js, speed)
            
            target_gear = 1
            for i, th in enumerate([0, 45, 90, 145, 200, 250]):
                if speed > th: target_gear = i + 1
            
            R.d['steer'], R.d['accel'], R.d['brake'], R.d['gear'] = steer, accel, brake, target_gear

            # --- REGISTRAZIONE ---
            if headers is None:
                headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
                for k in sorted(S.d.keys()):
                    if k in KEYS_TO_IGNORE: continue
                    val = S.d[k]
                    if isinstance(val, list): headers.extend([f"{k}_{i}" for i in range(len(val))])
                    else: headers.append(k)

            row = [time.time()-t0, steer, accel, brake, target_gear]
            for k in sorted(S.d.keys()):
                if k in KEYS_TO_IGNORE: continue
                val = S.d[k]
                if isinstance(val, list): row.extend(val)
                else: row.append(val)
            lap_buffer.append(row)

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nUscita.")
    finally:
        so.close()
        pygame.quit()

if __name__ == "__main__":
    manual_recording()



# --- LOGICA DI REGISTRAZIONE E VALIDAZIONE (SETTING F1) ---
# 1. SOGLIA CORDOLO INTEGRATA: trackPos tra 1.0 e 1.3 è considerato traiettoria valida.
#    L'auto può salire sui cordoli senza che il giro venga marcato come 'is_dirty'.
# 2. FUORI PISTA = RESET IMMEDIATO: Se abs(trackPos) > 1.3 (erba o sabbia), il giro 
#    viene annullato (is_dirty = True) e viene inviato il comando di RESTART al server.
# 3. GESTIONE DANNI: Qualsiasi incremento di 'damage' (urti contro muri o auto) 
#    marca il giro come 'sporco'. Il file non verrà salvato a fine giro, ma la 
#    gara continua senza reset automatico per permettere di fare pratica.
# 4. SALVATAGGIO AUTOMATICO: Ogni giro completato che non sia 'sporco' viene 
#    salvato su disco. Il nome del file include il tempo cronometrato (MM-SS-mmm).