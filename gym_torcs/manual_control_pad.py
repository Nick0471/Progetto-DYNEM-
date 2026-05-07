import pygame
import snakeoil3_jm2 as snakeoil3
import time
import json

class ArcadeController:
    def __init__(self):
        # Inizializza pygame e il modulo joystick
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[OK] Controller rilevato: {self.joystick.get_name()}")
        else:
            print("[ERRORE] Nessun controller rilevato! Assicurati che DS4Windows sia attivo.")

        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }
        
        # Variabili per evitare che la marcia cambi 10 volte con una sola pressione
        self.gear_up_pressed = False
        self.gear_down_pressed = False

    def update(self, sensors):
        # Aggiorna gli eventi di pygame per leggere gli input del controller
        pygame.event.pump()

        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)

        target_accel = 0.0
        target_brake = 0.0
        steer_input = 0.0

        if self.joystick:
            # ========================
            # LETTURA INPUT CONTROLLER (Mapping Xbox 360 / DS4Windows)
            # ========================
            # Asse 0: Levetta Sinistra X (Sinistra = -1.0, Destra = 1.0)
            # Asse 4: Grilletto Sinistro L2 (-1.0 rilasciato, 1.0 premuto)
            # Asse 5: Grilletto Destro R2 (-1.0 rilasciato, 1.0 premuto)
            
            # --- STEERING ---
            # Moltiplico per -0.6 per mantenere la stessa scala del tuo codice originale 
            # (dove Left era +0.6 e Right era -0.6)
            raw_steer = self.joystick.get_axis(0)
            if abs(raw_steer) > 0.1: # Deadzone iniziale della levetta
                steer_input = -raw_steer * 0.6 

            # --- ACCELERATORE (R2) ---
            # Pygame legge i grilletti da -1 a 1. Li convertiamo da 0 a 1.
            rt = self.joystick.get_axis(5)
            # Ignora valori falsi prima della prima pressione
            if rt > -0.99: 
                target_accel = (rt + 1.0) / 2.0

            # --- FRENO (L2) ---
            lt = self.joystick.get_axis(4)
            if lt > -0.99:
                target_brake = (lt + 1.0) / 2.0

            # --- MARCE (L1 / R1) ---
            # Tasto 4 = L1 (Scala marcia), Tasto 5 = R1 (Sali marcia)
            btn_up = self.joystick.get_button(0)
            btn_down = self.joystick.get_button(3)

            if btn_up and not self.gear_up_pressed:
                self.state['gear'] += 1
            self.gear_up_pressed = btn_up

            if btn_down and not self.gear_down_pressed:
                self.state['gear'] -= 1
            self.gear_down_pressed = btn_down

        # ========================
        # ACCELERAZIONE SMOOTH
        # ========================
        self.state['accel'] += (target_accel - self.state['accel']) * 0.1

        # ========================
        # FRENO SMOOTH
        # ========================
        self.state['brake'] += (target_brake - self.state['brake']) * 0.2

        # ========================
        # STEERING LOGIC (Originale)
        # ========================
        # LIMITE VELOCITÀ
        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_input *= max_steer

        # SE NON STAI STERZANDO → VAI DRITTO
        if abs(steer_input) < 0.01:
            steer_target = 0.0
        else:
            stability = angle * 0.3
            steer_target = steer_input - stability

        # SMOOTH
        self.state['steer'] += (steer_target - self.state['steer']) * 0.2

        # DEAD ZONE
        if abs(self.state['steer']) < 0.02:
            self.state['steer'] = 0.0

        # CLAMPING
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))

        # MARCE LIMITI
        self.state['gear'] = max(-1, min(6, self.state['gear']))


# ============================================================
# MAIN
# ============================================================

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()

    client.get_servers_input()

    print("Arcade driving mode attivo con CONTROLLER")
    print("Levetta SX per sterzare, R2 Accelera, L2 Frena, R1/L1 per le marce")

    # CSV log
    log_csv = open("manual_log.csv", "w")
    log_csv.write("time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage\n")

    # JSON log (step-by-step strutturato)
    log_json = []
    
    t0 = time.time()
    step = 0

    while True:
        S = client.S.d

        controller.update(S)
        a = controller.state
        
        print(f"steer={a['steer']:.2f} accel={a['accel']:.2f} brake={a['brake']:.2f} gear={a['gear']}")

        client.R.d['steer'] = a['steer']
        client.R.d['accel'] = a['accel']
        client.R.d['brake'] = a['brake']
        client.R.d['gear'] = a['gear']
        client.R.d['clutch'] = 0.0
        client.R.d['meta'] = 0

        client.respond_to_server()
        client.get_servers_input()

        current_time = time.time() - t0

        # ===== CSV LOG =====
        log_csv.write(
            f"{current_time},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
            f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
            f"{S.get('rpm',0)},{S.get('damage',0)}\n"
        )

        # ===== JSON LOG (STEP-BY-STEP) =====
        log_json.append({
            "step": step,
            "time": current_time,
            "action": {
                "steer": a['steer'],
                "accel": a['accel'],
                "brake": a['brake'],
                "gear": a['gear']
            },
            "state": {
                "speedX": S.get('speedX', 0),
                "trackPos": S.get('trackPos', 0),
                "angle": S.get('angle', 0),
                "rpm": S.get('rpm', 0),
                "damage": S.get('damage', 0)
            }
        })

        step += 1

        # salva JSON ogni tot step (evita perdita dati)
        if step % 100 == 0:
            with open("manual_log.json", "w") as f:
                json.dump(log_json, f, indent=2)
                
        time.sleep(0.02)

if __name__ == "__main__":
    main()