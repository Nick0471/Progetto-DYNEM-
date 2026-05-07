import socket
import sys
import getopt
import os
import time
import csv # Aggiunto per il salvataggio del dataset

PI = 3.14159265359
data_size = 2**17

ophelp =  'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage = usage + ophelp
version = "20130505-2-DatasetCollection"

def clip(v, lo, hi):
    if v < lo: return lo
    elif v > hi: return hi
    else: return v

def bargraph(x, mn, mx, w, c='X'):
    if not w: return ''
    if x < mn: x = mn
    if x > mx: x = mx
    tx = mx - mn
    if tx <= 0: return 'backwards'
    upw = tx / float(w)
    if upw <= 0: return 'what?'
    negpu, pospu, negnonpu, posnonpu = 0, 0, 0, 0
    if mn < 0:
        if x < 0:
            negpu = -x + min(0, mx)
            negnonpu = -mn + x
        else:
            negnonpu = -mn + min(0, mx)
    if mx > 0:
        if x > 0:
            pospu = x - max(0, mn)
            posnonpu = mx - x
        else:
            posnonpu = mx - max(0, mn)
    nnc = int(negnonpu / upw) * '-'
    npc = int(negpu / upw) * c
    ppc = int(pospu / upw) * c
    pnc = int(posnonpu / upw) * '_'
    return '[%s]' % (nnc + npc + ppc + pnc)

class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False):
        self.vision = vision
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.maxEpisodes = 1
        self.trackname = 'unknown'
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000
        self.parse_the_command_line()
        if H: self.host = H
        if p: self.port = p
        if i: self.sid = i
        if e: self.maxEpisodes = e
        if t: self.trackname = t
        if s: self.stage = s
        if d: self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        
        # --- VARIABILI PER IL DATASET ---
        self.dataset_records = []
        self.laps_completed = 0
        self.best_lap_time = float('inf')
        self.last_lap_time_seen = 0.0
        
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                       ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug = True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host = opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid = opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname = opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage = int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port = int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes = int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps = int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        if not self.so: return
        sockdata = str()

        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. " +
                        "You were in %d place.") %
                        (self.port, self.S.d.get('racePos', -1))))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")
                    print(self.S)
                break

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    # --- NUOVA FUNZIONE: RACCOLTA DATI ---
    def record_data(self):
        if not self.S.d: return
        
        # Aggiornamento contatori giro e miglior tempo
        current_last_lap = self.S.d.get('lastLapTime', 0.0)
        if current_last_lap > 0 and current_last_lap != self.last_lap_time_seen:
            self.laps_completed += 1
            if current_last_lap < self.best_lap_time:
                self.best_lap_time = current_last_lap
            self.last_lap_time_seen = current_last_lap

        # Estrazione Feature Sensori (State)
        row = {
            'speedX': self.S.d.get('speedX', 0),
            'speedY': self.S.d.get('speedY', 0),
            'speedZ': self.S.d.get('speedZ', 0),
            'angle': self.S.d.get('angle', 0),
            'rpm': self.S.d.get('rpm', 0),
            'trackPos': self.S.d.get('trackPos', 0),
            'z': self.S.d.get('z', 0),
            'distFromStart': self.S.d.get('distFromStart', 0),
            'damage': self.S.d.get('damage', 0)
        }
        
        # Appiattimento Array Sensori Pista (19 sensori)
        track_sensors = self.S.d.get('track', [0]*19)
        for i, val in enumerate(track_sensors):
            row[f'track_{i}'] = val
            
        # Appiattimento Array Wheel Spin Velocity (4 ruote)
        wheel_spin = self.S.d.get('wheelSpinVel', [0]*4)
        for i, val in enumerate(wheel_spin):
            row[f'wheelSpinVel_{i}'] = val
            
        # Estrazione Label Azioni (Actions)
        row['cmd_steer'] = self.R.d.get('steer', 0)
        row['cmd_accel'] = self.R.d.get('accel', 0)
        row['cmd_brake'] = self.R.d.get('brake', 0)
        row['cmd_gear'] = self.R.d.get('gear', 0)
        row['cmd_clutch'] = self.R.d.get('clutch', 0)
        
        self.dataset_records.append(row)

    # --- NUOVA FUNZIONE: SALVATAGGIO SU CSV ---
    def save_dataset(self):
        if not self.dataset_records:
            print("Nessun dato registrato. Il dataset non verrà salvato.")
            return
            
        # Creazione cartella "dataset"
        save_dir = "dataset"
        os.makedirs(save_dir, exist_ok=True)
        
        # Formattazione nome file
        best_time_str = f"{self.best_lap_time:.2f}".replace('.', '_') if self.best_lap_time != float('inf') else "no_time"
        filename = f"{self.trackname}_{self.laps_completed}laps_{best_time_str}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # Salvataggio CSV
        try:
            with open(filepath, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.dataset_records[0].keys())
                writer.writeheader()
                writer.writerows(self.dataset_records)
            print(f"\n[+] DATASET SALVATO CON SUCCESSO: {filepath}")
            print(f"    Totale righe acquisite: {len(self.dataset_records)}\n")
        except Exception as e:
            print(f"Errore durante il salvataggio del dataset: {e}")

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d." % (self.maxSteps, self.port)))
        self.save_dataset() # Richiama il salvataggio prima di chiudere la connessione
        self.so.close()
        self.so = None

class ServerState():
    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

class DriverAction():
    def __init__(self):
       self.actionstr = str()
       self.d = { 'accel':0.2, 'brake':0, 'clutch':0, 'gear':1, 'steer':0, 'focus':[-90,-45,0,45,90], 'meta':0 }

    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0,1]:
            self.d['meta'] = 0
        if type(self.d['focus']) is not list or min(self.d['focus']) < -180 or max(self.d['focus']) > 180:
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out

def destringify(s):
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

# ================= USER CONFIGURABLE PARAMETERS =================
TARGET_SPEED = 160
STEER_GAIN = 30
CENTERING_GAIN = 0.2
BRAKE_THRESHOLD = 0.4
GEAR_SPEEDS = [0, 50, 80, 120, 150, 200]
ENABLE_TRACTION_CONTROL = True

SAFE_GENTLE_CORNER_SPEED = 140
SAFE_SHARP_CORNER_SPEED = 65
TARGET_STRAIGHT_SPEED = 194
CORNER_READING = 2.0
SLOW_DOWN_DISTANCE = 60
STRAIGHT_DISTANCE = 120
BRAKING_INTENSITY = 0.3
STEERING_EFFECT = 1.6

def get_min_sensor_data(S):
    left_sensors = S['track'][:9]
    right_sensors = S['track'][10:]
    return min(min(left_sensors), min(right_sensors))

def is_corner(S, min_reading):
    if min_reading < CORNER_READING or S['track'][9] < S['speedX'] * 0.65:
        return True
    return False

def is_straight(current_speed, forward_length):
    if current_speed >= (TARGET_SPEED - 5) and forward_length > STRAIGHT_DISTANCE:
        return True
    return False

def hold_acceleration(S, safe_speed):
    if is_corner(S, get_min_sensor_data(S)) and S['speedX'] > safe_speed:
        return True
    return False

def slow_down(S):
    if max(S['track'][7:12]) < S['speedX'] * 0.60:
        return True
    return False

def calculate_corner_speed(S):
    if max(S['track'][8:11]) < SLOW_DOWN_DISTANCE:
        return SAFE_SHARP_CORNER_SPEED
    return SAFE_GENTLE_CORNER_SPEED

def calculate_steering(S):
    steer = (S['angle'] * STEER_GAIN / PI) - (S['trackPos'] * CENTERING_GAIN)
    if is_corner(S, get_min_sensor_data(S)):
        bias = (sum(S['track'][10:]) / 8) - (sum(S['track'][:9]) / 8)
        if bias < 0:
            steer += 0.46
        elif bias > 0:
            steer -= 0.46
    return max(-1, min(1, steer))

def calculate_throttle(S, R):
    target_speed = TARGET_STRAIGHT_SPEED if is_straight(S['speedX'], S['track'][9]) else TARGET_SPEED
    if S['speedX'] < target_speed - (R['steer'] * STEERING_EFFECT):
        accel = min(1.0, R['accel'] + 0.4)
    else:
        accel = max(0.0, R['accel'] - 0.2)
    if hold_acceleration(S, calculate_corner_speed(S)):
        accel = max(0.0, R['accel'] - 0.2)
    if S['speedX'] < 10:
        accel = 1.0
    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    brake = BRAKING_INTENSITY if abs(S['angle']) > BRAKE_THRESHOLD else 0.0
    if slow_down(S):
        brake += 0.1
    return min(1.0, brake)
    
def shift_gears(S):
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S['speedX'] > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL and ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])) > 2:
        accel -= 0.1
    return max(0.0, accel)

def drive_modular(c):
    S, R = c.S.d, c.R.d
    R['steer'] = calculate_steering(S)
    R['accel'] = calculate_throttle(S, R)
    R['brake'] = apply_brakes(S)
    R['accel'] = traction_control(S, R['accel'])
    R['gear'] = shift_gears(S)

if __name__ == "__main__":
    C = Client(p=3001)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_modular(C)
        C.record_data()  # <--- Registra lo stato e l'azione ad ogni step
        C.respond_to_server()
    C.shutdown()